from copy import deepcopy
from pyquaternion import Quaternion
import numpy as np


def interpolate(key0, key1, t=0.5):
    mesh = deepcopy(key0)

    # TODO: Takes too long
    print("Interpolating IICs")
    for eid, fids in enumerate(mesh.edge2face):
        left = fids[0]
        right = fids[1]
        if left is None or right is None:
            continue

        D0 = np.matmul(
            key0.tangent_frames[left], np.linalg.inv(key0.tangent_frames[right])
        )
        D1 = np.matmul(
            key1.tangent_frames[left], np.linalg.inv(key1.tangent_frames[right])
        )

        q0 = Quaternion(matrix=D0)
        q1 = Quaternion(matrix=D1)

        q = Quaternion.slerp(q0, q1, amount=t)

        m = q.rotation_matrix
        mesh.Q[eid] = m
        mesh.L[eid] = (1 - t) * key0.L[eid] + t * key1.L[eid]

    D0 = key0.tangent_frames[0]
    D1 = key1.tangent_frames[0]
    q0 = Quaternion(matrix=D0)
    q1 = Quaternion(matrix=D1)
    q = Quaternion.slerp(q0, q1, amount=t)
    mesh.tangent_frames[0] = q.rotation_matrix
    mesh.verts[0] = (1 - t) * (key0.verts[0]) + t * (key1.verts[0])

    print("Solving the face system.")
    mesh._compute_face_system()
    mesh._prefactor_face_system()
    mesh._solve_face_system()

    print("Solving the vertex system.")
    mesh._compute_vertex_system()
    mesh._prefactor_vertex_system()
    mesh._solve_vertex_system()

    mesh.update()

    return mesh


def evaluate(mesh, key0, key1, t):
    E = mesh.edges
    l0 = np.sqrt(np.sum((key0.verts[E.T[0]] - key0.verts[E.T[1]])**2, axis=1))
    l1 = np.sqrt(np.sum((key1.verts[E.T[0]] - key1.verts[E.T[1]])**2, axis=1))
    l_true = (1-t)*l0 + t*l1
    l_pred = np.sqrt(np.sum((mesh.verts[E.T[0]] - mesh.verts[E.T[1]])**2, axis=1))
    return np.abs(l_pred - l_true)/l_true


def refine(mesh):
    NF = mesh.faces.shape[0]
    mesh._compute_vertex_system()
    mesh._prefactor_vertex_system()
    
    mesh._compute_face_system(constrained=range(NF))
    mesh._prefactor_face_system()
    mesh._solve_face_system(constrained=range(NF))

    mesh._solve_vertex_system()
    
    return mesh