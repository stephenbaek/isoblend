import open3d
import numpy as np
import scipy
from scipy.sparse import coo_matrix
import scipy.sparse.linalg
from copy import deepcopy


class IICMesh:
    least_squares_weight = 1.0

    def __init__(self, filename=None):
        self.mesh = None
        self.verts = []
        self.faces = []
        self.edges = []
        self.edge2face = []
        self.tangent_frames = []
        self.L = None  # lengths in IIC
        self.Q = None  # the angles in IIC. (improvement from the paper: use quaternion)
        self.H = None  # Face system
        self.G = None  # Vertex system

        if filename is not None:
            self.read(filename)

    def read(self, filename):
        """Reads a triangular mesh from a file (*.obj, *.ply, etc.)

        Args:
            filename: string file name
        """
        self.mesh = open3d.io.read_triangle_mesh(filename)
        self.mesh.compute_triangle_normals()
        if not self.mesh.is_edge_manifold():
            raise Exception("Non-manifold mesh detected.")
        self._compute_topology_indices()
        self._compute_local_tangent_frames()
        self._compute_IIC()
        self._compute_face_system()
        self._compute_vertex_system()

    def show(self, zoom=1, lookat=[0, 0, 0], front=[-1, 0.3, -1], up=[0, 1, 0]):
        """Displays the mesh. A new window will pop up.

        TODO: Inline display for Jupyter notebooks?

        Args:
            zoom: smaller is closer to the object
            front: front vector of the camera
            lookat: lookat vector of the camera (where the camera is focused at)
            up: up vector of the camera
        """
        open3d.visualization.draw_geometries(
            [self.mesh], zoom=zoom, front=front, lookat=lookat, up=up
        )

    def _compute_topology_indices(self):
        """Compute topology indices."""
        self.verts = np.asarray(self.mesh.vertices)
        self.faces = np.asarray(self.mesh.triangles)

        I0 = self.faces[:, 0]
        I1 = self.faces[:, 1]
        I2 = self.faces[:, 2]

        # edges
        E = np.concatenate(
            (
                np.concatenate((np.expand_dims(I0, 1), np.expand_dims(I1, 1)), axis=1),
                np.concatenate((np.expand_dims(I1, 1), np.expand_dims(I2, 1)), axis=1),
                np.concatenate((np.expand_dims(I2, 1), np.expand_dims(I0, 1)), axis=1),
            ),
            axis=0,
        )

        # Edges will be stored as [i, j] where i, j are vertex indices (i < j).
        # When edge (i, j) is part of triangle (p, q, r),
        # the triangle is called a "left triangle" if (i, j) appears in the even permutation of (p, q, r)
        # otherwise, "right triangle"
        perm = np.argsort(E, axis=1)[:, 0]  # permutation of the edges
        E = np.sort(E, axis=1)  # make it i < j
        [E, F2E] = np.unique(
            E, axis=0, return_inverse=True
        )  # remove duplicated entries

        # Map from edge index to the indices of the parent triangles. First element is left triangle.
        # Second element is right triangle. None if boundary edge.
        E2F = [[None, None] for _ in range(E.shape[0])]
        for f, e in enumerate(F2E):
            E2F[e][perm[f]] = f % self.faces.shape[0]

        self.edges = E
        self.edge2face = E2F
        self.face2edge = np.reshape(F2E, (3, self.faces.shape[0])).T

    def _compute_local_tangent_frames(self):
        """Builds local tangent space frames at the triangles"""
        I0 = self.faces[:, 0]
        I1 = self.faces[:, 1]

        N = np.asarray(self.mesh.triangle_normals)
        X = self.verts[I1] - self.verts[I0]
        X /= np.sqrt(np.sum(X ** 2, axis=1, keepdims=True))  # normalize
        Y = np.cross(N, X)

        # tangent frames (a |F| x 3 x 3 tensor)
        self.tangent_frames = np.concatenate(
            (
                np.expand_dims(X, axis=2),
                np.expand_dims(Y, axis=2),
                np.expand_dims(N, axis=2),
            ),
            axis=2,
        )

    def _compute_IIC(self):
        """Builds the isometry-invariant intrinsic coordinates (IICs).

        See Section 2 of the paper.
        """
        self.L = np.zeros((self.edges.shape[0],))
        self.Q = np.zeros((self.edges.shape[0], 3, 3))
        for eid, e in enumerate(self.edges):
            left = self.edge2face[eid][0]
            right = self.edge2face[eid][1]
            if left is not None and right is not None:
                self.Q[eid] = np.matmul(
                    self.tangent_frames[left], np.linalg.inv(self.tangent_frames[right])
                )
                self.L[eid] = np.sqrt(
                    np.sum((self.verts[e[0]] - self.verts[e[1]]) ** 2)
                )

    def _compute_face_system(self, constrained=[0]):
        """Constructs a system of equation for reconstructing faces from IICs."""
        NE = self.edges.shape[0]
        NF = self.faces.shape[0]

        i = []
        j = []
        val = []

        for eid, fids in enumerate(self.edge2face):
            left = fids[0]
            right = fids[1]

            # TODO: matrix becomes singular when boundary exists
            if left is None or right is None:
                continue

            i.extend([3 * eid, 3 * eid + 1, 3 * eid + 2])
            j.extend([3 * left, 3 * left + 1, 3 * left + 2])
            val.extend([-1, -1, -1])

            i.extend([3 * eid, 3 * eid, 3 * eid])
            j.extend([3 * right, 3 * right + 1, 3 * right + 2])
            val.extend([self.Q[eid][0][0], self.Q[eid][0][1], self.Q[eid][0][2]])

            i.extend([3 * eid + 1, 3 * eid + 1, 3 * eid + 1])
            j.extend([3 * right, 3 * right + 1, 3 * right + 2])
            val.extend([self.Q[eid][1][0], self.Q[eid][1][1], self.Q[eid][1][2]])

            i.extend([3 * eid + 2, 3 * eid + 2, 3 * eid + 2])
            j.extend([3 * right, 3 * right + 1, 3 * right + 2])
            val.extend([self.Q[eid][2][0], self.Q[eid][2][1], self.Q[eid][2][2]])

        # Constrained faces
        for c, fid in enumerate(constrained):
            i.extend([3 * NE + 3 * c, 3 * NE + 3 * c + 1, 3 * NE + 3 * c + 2])
            j.extend([3*fid, 3*fid+1, 3*fid + 2])
            val.extend(
                [
                    self.least_squares_weight,
                    self.least_squares_weight,
                    self.least_squares_weight,
                ]
            )

        self.H = coo_matrix((val, (i, j)), shape=(3 * NE + 3 * len(constrained), 3 * NF))

    def _compute_vertex_system(self):
        # Vertex system
        NV = self.verts.shape[0]
        NF = self.faces.shape[0]

        i = []
        j = []
        val = []

        for fid, f in enumerate(self.faces):
            i.extend(
                [3 * fid, 3 * fid, 3 * fid + 1, 3 * fid + 1, 3 * fid + 2, 3 * fid + 2]
            )
            j.extend([f[0], f[1], f[1], f[2], f[2], f[0]])
            val.extend([-1, 1, -1, 1, 1, -1])

        i.extend([3 * NF])
        j.extend([0])
        val.extend([self.least_squares_weight])
        self.G = coo_matrix((val, (i, j)), shape=(3 * NF + 1, NV))

    def _prefactor_face_system(self):
        self.HTHinv = scipy.sparse.linalg.splu((self.H.T * self.H).tocsr())

    def _prefactor_vertex_system(self):
        self.GTGinv = scipy.sparse.linalg.splu((self.G.T * self.G).tocsr())

    def _solve_face_system(self, constrained=[0]):
        NE = self.edges.shape[0]
        rhs = np.zeros((3 * NE + 3*len(constrained), 3))
        for i, fid in enumerate(constrained):
            rhs[3 * NE + 3 * i, 0] = self.least_squares_weight * self.tangent_frames[fid][0][0]
            rhs[3 * NE + 3 * i, 1] = self.least_squares_weight * self.tangent_frames[fid][0][1]
            rhs[3 * NE + 3 * i, 2] = self.least_squares_weight * self.tangent_frames[fid][0][2]
            rhs[3 * NE + 3 * i + 1, 0] = self.least_squares_weight * self.tangent_frames[fid][1][0]
            rhs[3 * NE + 3 * i + 1, 1] = self.least_squares_weight * self.tangent_frames[fid][1][1]
            rhs[3 * NE + 3 * i + 1, 2] = self.least_squares_weight * self.tangent_frames[fid][1][2]
            rhs[3 * NE + 3 * i + 2, 0] = self.least_squares_weight * self.tangent_frames[fid][2][0]
            rhs[3 * NE + 3 * i + 2, 1] = self.least_squares_weight * self.tangent_frames[fid][2][1]
            rhs[3 * NE + 3 * i + 2, 2] = self.least_squares_weight * self.tangent_frames[fid][2][2]

        x = self.HTHinv.solve(self.H.T * rhs)
        self.x = x  # for debugging
        self.old = deepcopy(self.tangent_frames)  # for debugging

        # Orthogonalize
        # TODO: compare speed between qr and orth
        for fid, f in enumerate(self.faces):
            # self.tangent_frames[fid] = scipy.linalg.orth(x[3*fid:3*fid+3, :])
            Q, R = np.linalg.qr(x[3 * fid : 3 * fid + 3, :])
            dR = np.array([1, 1, 1])
            for i in range(3):
                if R[i][i] < 0:
                    dR[i] = -1
            self.tangent_frames[fid] = Q * dR

    def _solve_vertex_system(self):
        NF = self.faces.shape[0]
        rhs = np.zeros((3 * NF + 1, 3))
        for fid, f in enumerate(self.faces):
            lengths = self.L[self.face2edge[fid]]
            verts = self.verts[f]
            v01 = verts[1] - verts[0]
            v01 /= np.sqrt(np.sum(v01 ** 2))
            v02 = verts[2] - verts[0]
            v02 /= np.sqrt(np.sum(v02 ** 2))
            v21 = verts[1] - verts[2]
            v21 /= np.sqrt(np.sum(v21 ** 2))
            cos0 = np.dot(v01, v02)
            cos1 = np.dot(v21, v01)
            sin0 = np.sqrt(1 - cos0 * cos0)
            sin1 = np.sqrt(1 - cos1 * cos1)

            X = self.tangent_frames[fid].T[0]
            Y = self.tangent_frames[fid].T[1]

            rhs[3 * fid, :] = lengths[0] * X
            rhs[3 * fid + 1, :] = -lengths[1] * cos1 * X + lengths[1] * sin1 * Y
            rhs[3 * fid + 2, :] = lengths[2] * cos0 * X + lengths[2] * sin0 * Y

        rhs[3 * NF, 0] = self.least_squares_weight * self.verts[0][0]
        rhs[3 * NF, 1] = self.least_squares_weight * self.verts[0][1]
        rhs[3 * NF, 2] = self.least_squares_weight * self.verts[0][2]

        self.verts = self.GTGinv.solve(self.G.T * rhs)

    def update(self):
        for vid, v in enumerate(self.verts):
            self.mesh.vertices[vid][0] = v[0]
            self.mesh.vertices[vid][1] = v[1]
            self.mesh.vertices[vid][2] = v[2]
        self.mesh.compute_triangle_normals()
