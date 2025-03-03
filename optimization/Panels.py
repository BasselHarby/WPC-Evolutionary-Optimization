import multiprocessing
import os
import sys

FREECADPATH = 'C:\\Program Files\\FreeCAD 1.0\\bin'
sys.path.append(FREECADPATH)

import math
import numpy as np
import FreeCAD
import Part
import ObjectsFem
import Fem
from femtools import ccxtools
import Draft
from scipy import signal
from FreeCAD import Units


def Tmeshrefiner(pts, mesh_size):
    x = pts[:, 1]
    y = pts[:, 2]
    dx, dy = x[+1:] - x[:-1], y[+1:] - y[:-1]
    ds = np.array((0, *np.sqrt(dx * dx + dy * dy)))
    s = np.cumsum(ds)
    narr = np.floor(np.diff(s) / mesh_size).astype(int)
    newXs = augment_with_divisions(s, narr)
    xinter = np.round(np.interp(newXs, s, x), 2)
    yinter = np.round(np.interp(newXs, s, y), 2)
    x = np.zeros_like(xinter)
    return np.array([x, xinter, yinter]).transpose()


def augment_with_divisions(array, divisions):
    augmented_array = []
    for i in range(len(array) - 1):
        # Create `divisions[i] + 1` points between `array[i]` and `array[i+1]`
        augmented_array.extend(np.linspace(array[i], array[i + 1], divisions[i] + 2)[:-1])
    # Add the last element of the original array
    augmented_array.append(array[-1])
    return np.array(augmented_array)


class TrPanel:
    def __init__(self, doc, length, width, thickness, amplitude, period, phi, meshsize):
        self.doc = doc
        self.length = length
        self.width = width
        self.thickness = thickness
        self.amplitude = amplitude
        self.period = period
        self.phi = phi
        self.meshsize = meshsize
        self.num_divisions_width = int(self.width / self.meshsize)
        self.analysis_object = ObjectsFem.makeAnalysis(doc, "Analysis")
        self.createPanel()  # 1
        self.FEMinit()  # 2
        self.createFEMmesh()  # 3
        self.runAnalysis()  # 4

    def zigzag(self, y):  # Returns the height at a certain y value
        return np.tan(np.radians(self.phi)) * self.period * signal.sawtooth(2 * np.pi * y / self.period, width=0.5) / 4.

    def createPanel(self):
        y = np.arange(0, self.length + 0.001, 0.01)
        x = np.zeros_like(y)
        z = self.zigzag(y)
        z = np.round(z, 2)
        maxpt = max(z)

        # check if panel is trapezoidal or traingular
        if self.amplitude / 2 < maxpt:
            z[z > self.amplitude / 2.] = self.amplitude / 2.
            z[z < -self.amplitude / 2.] = -self.amplitude / 2.
            z = z + self.amplitude / 2.
            maxpt = max(z)
        else:
            z = z + maxpt
            maxpt = max(z)
        pts = np.array([x, y, z]).transpose()
        # filter slanted part
        pts = pts[((pts[:, 2] == maxpt) | (pts[:, 2] == 0))]

        # filter flat part
        result = []
        prev_y = None
        start = pts[0]
        for i in range(len(pts)):
            if prev_y is None:
                prev_y = pts[i][2]
                start = pts[i]
            elif pts[i][2] != prev_y:
                result.append(start)
                result.append(pts[i - 1])
                start = pts[i]
                prev_y = pts[i][2]

        result.append(start)
        result.append(pts[-1])
        pts = np.array(result)

        pts = Tmeshrefiner(pts, self.meshsize)
        pts = np.unique(pts, axis=0)  # remove duplicate points if any
        self.pts = pts
        point_list = [FreeCAD.Vector(itm) for itm in pts]

        W = Draft.make_wire(point_list)
        self.part = Draft.extrude(W, FreeCAD.Vector(self.width, 0, 0))
        self.part = self.doc.getObject('Extrusion')
        self.doc.recompute()

    def FEMinit(self):
        # Solver object
        self.solver_object = ObjectsFem.makeSolverCalculiXCcxTools(self.doc, "CalculiX")
        self.solver_object.GeometricalNonlinearity = 'linear'
        self.solver_object.ThermoMechSteadyState = True
        self.solver_object.MatrixSolverType = 'default'
        self.solver_object.IterationsControlParameterTimeUse = False
        self.analysis_object.addObject(self.solver_object)
        # Material Object
        E = 3345  # MPa (3.5 GPa)
        nu = 0.3
        rho = 540  # kg/m^3
        material_object = ObjectsFem.makeMaterialSolid(self.doc, "SolidMaterial")
        mat = material_object.Material
        mat['Name'] = "WPC"
        mat['YoungsModulus'] = str(E) + " MPa"
        mat['PoissonRatio'] = "0.30"
        mat['Density'] = str(rho) + " kg/m^3"
        material_object.Material = mat
        self.analysis_object.addObject(material_object)
        # Self Weight Constraint
        con_selfweight = ObjectsFem.makeConstraintSelfWeight(self.doc, "ConstraintSelfWeight")
        self.analysis_object.addObject(con_selfweight)
        # shell thickness
        self.thickness_obj = ObjectsFem.makeElementGeometry2D(self.doc, self.thickness, "Thickness")
        self.analysis_object.addObject(self.thickness_obj)
        # Load Line Direction
        sh_load_line = Part.makeLine(FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(0, 0, 10))
        load_line = self.doc.addObject("Part::Feature", "Load_direction_line")
        load_line.Shape = sh_load_line
        self.loadDirection = load_line
        # mesh
        self.femmesh_obj = self.doc.addObject('Fem::FemMeshObject', 'RectMesh')
        self.analysis_object.addObject(self.femmesh_obj)

        # constraints displacement
        supportsL, supportsR = self.getsupportReference()
        self.supportL = ObjectsFem.makeConstraintDisplacement(self.doc, "supportsL")
        self.supportL.yDisplacement = 0
        self.supportL.yFree = False
        self.supportL.zDisplacement = 0
        self.supportL.zFree = False
        self.analysis_object.addObject(self.supportL)
        self.supportL.References = supportsL

        self.supportR = ObjectsFem.makeConstraintDisplacement(self.doc, "supportsR")
        self.supportR.xDisplacement = 0
        self.supportR.xFree = False
        self.supportR.yDisplacement = 0
        self.supportR.yFree = False
        self.supportR.zDisplacement = 0
        self.supportR.zFree = False
        self.analysis_object.addObject(self.supportR)
        self.supportR.References = supportsR

        # Loads
        shape_faces = self.part.Shape.Faces
        self.Area = 0
        for i, Face in enumerate(shape_faces):
            LoadFaces = (self.part, "Face" + str(i + 1))
            Load = Face.Area * 0.0006
            self.Area += Face.Area
            force_constraint = ObjectsFem.makeConstraintForce(self.doc, "Force" + str(i + 1))
            force_constraint.References = LoadFaces
            force_constraint.Direction = (self.loadDirection, ["Edge1"])
            force_constraint.Force = str(Load) + " N"
            force_constraint.Reversed = True
            self.analysis_object.addObject(force_constraint)

    def createFEMmesh(self):
        femmesh = Fem.FemMesh()
        node_id = 1
        n = int(np.round(self.width / self.meshsize, 0))
        xList = np.linspace(0, self.width, n)  # in the width direction
        yList = self.pts[:, 1]  # in the length direction
        zlist = self.pts[:, 2]
        for x in xList:
            for i, y in enumerate(yList):
                z = zlist[i]
                femmesh.addNode(x, y, z, node_id)
                node_id += 1

        num_divisions_width = len(xList) - 1
        num_divisions_length = len(yList) - 1
        for i in range(num_divisions_width):
            for j in range(num_divisions_length):
                n1 = i * (num_divisions_length + 1) + j + 1
                n2 = n1 + 1
                n3 = n1 + num_divisions_length + 2
                n4 = n1 + num_divisions_length + 1
                femmesh.addQuad(n4, n3, n2, n1)

        self.femmesh_obj.FemMesh = femmesh

    def getsupportReference(self):
        shape_edges = self.part.Shape.Edges
        lowest_z = min(np.transpose(self.pts)[-1])
        support_edgesL = []
        support_edgesR = []
        for i, edge in enumerate(shape_edges):
            if (edge.Vertexes[0].X == 0 and edge.Vertexes[1].X == 0 and (
                    math.isclose(edge.Vertexes[0].Z, lowest_z, rel_tol=0.05) or math.isclose(edge.Vertexes[1].Z,
                                                                                             lowest_z, rel_tol=0.05))):
                support_edgesL.append((self.part, "Edge" + str(i + 1)))
            elif (edge.Vertexes[0].X == self.width and edge.Vertexes[1].X == self.width and (
                    math.isclose(edge.Vertexes[0].Z, lowest_z, rel_tol=0.05) or math.isclose(edge.Vertexes[1].Z,
                                                                                             lowest_z, rel_tol=0.05))):
                support_edgesR.append((self.part, "Edge" + str(i + 1)))
        return (support_edgesL, support_edgesR)

    def get_displacement_within_point(self, result_obj, target_x, target_y, radius):
        Nodes_vectors_list = list(result_obj.Mesh.FemMesh.Nodes.values())
        displacements_vectors_list = result_obj.DisplacementVectors
        nodes_array = np.array([(node.x, node.y, node.z) for node in Nodes_vectors_list])
        displacements_array = np.array([(disp.x, disp.y, disp.z) for disp in displacements_vectors_list])
        distances = np.sqrt((nodes_array[:, 0] - target_x) ** 2 + (nodes_array[:, 1] - target_y) ** 2)
        within_radius_indices = np.where(distances <= radius)[0]
        z_displacements = displacements_array[within_radius_indices, 2]
        return max(abs(z_displacements))

    def runAnalysis(self):
        self.fea = ccxtools.FemToolsCcx(solver=self.solver_object)
        self.fea.purge_results()
        self.fea.update_objects()
        self.fea.run()
        self.results = self.doc.getObject('CCX_Results')
        if self.results:
            self.max_disp = self.get_displacement_within_point(self.results, self.width / 2, self.length / 2, 100)
            self.maxStress = np.percentile(self.results.PrincipalMax, 99.5)
            self.volume = (self.Area * self.thickness) / 1000000000
        else:
            self.max_disp = 100000
            self.maxStress = 1000000
            self.volume = 1000000


# ******************************************************************************************

def meshrefiner(pts, mesh_size):
    x = pts[:, 1]
    y = pts[:, 2]
    dx, dy = x[+1:] - x[:-1], y[+1:] - y[:-1]
    ds = np.array((0, *np.sqrt(dx * dx + dy * dy)))
    s = np.cumsum(ds)
    nPoints = int(np.ceil(s[-1] / mesh_size))
    newXs = np.linspace(0, s[-1], nPoints)
    xinter = np.interp(newXs, s, x)
    yinter = np.interp(newXs, s, y)
    x = np.zeros_like(xinter)
    return np.array([x, xinter, yinter]).transpose()


class SinePanel:
    def __init__(self, doc, length, width, thickness, amplitude, period, meshsize):
        self.doc = doc
        self.length = length  # y direction
        self.width = width  # x direction
        self.thickness = thickness
        self.amplitude = amplitude
        self.period = period
        self.meshsize = meshsize
        self.num_divisions_width = int(self.width / self.meshsize)
        self.analysis_object = ObjectsFem.makeAnalysis(doc, "Analysis")
        self.createPanel()  # 1
        self.FEMinit()  # 2
        self.createFEMmesh()  # 3
        self.runAnalysis()  # 4

    def sin(self, y):  # Returns the height at a certain y value
        return self.amplitude * np.sin(2 * np.pi * y / self.period)

    def createPanel(self):
        y = np.arange(0, self.length + 0.001, 0.01)
        x = np.zeros_like(y)
        z = self.sin(y)
        z = np.round(z, 2)
        pts = np.array([x, y, z]).transpose()

        pts = meshrefiner(pts, self.meshsize)
        pts = np.unique(pts, axis=0)  # remove duplicate points if any
        self.pts = pts
        point_list = [FreeCAD.Vector(itm) for itm in pts]

        W = Draft.make_wire(point_list)
        self.part = Draft.extrude(W, FreeCAD.Vector(self.width, 0, 0))
        self.part = self.doc.getObject('Extrusion')
        self.doc.recompute()

    def FEMinit(self):
        # Solver object
        self.solver_object = ObjectsFem.makeSolverCalculiXCcxTools(self.doc, "CalculiX")
        self.solver_object.GeometricalNonlinearity = 'linear'
        self.solver_object.ThermoMechSteadyState = True
        self.solver_object.MatrixSolverType = 'default'
        self.solver_object.IterationsControlParameterTimeUse = False
        self.analysis_object.addObject(self.solver_object)
        # Material Object
        E = 3345  # MPa (3.5 GPa)
        nu = 0.3
        rho = 540  # kg/m^3
        material_object = ObjectsFem.makeMaterialSolid(self.doc, "SolidMaterial")
        mat = material_object.Material
        mat['Name'] = "WPC"
        mat['YoungsModulus'] = "3345 MPa"
        mat['PoissonRatio'] = "0.30"
        mat['Density'] = str(rho) + " kg/m^3"
        material_object.Material = mat
        self.analysis_object.addObject(material_object)
        # Self Weight Constraint
        con_selfweight = ObjectsFem.makeConstraintSelfWeight(self.doc, "ConstraintSelfWeight")
        self.analysis_object.addObject(con_selfweight)
        # shell thickness
        self.thickness_obj = ObjectsFem.makeElementGeometry2D(self.doc, self.thickness, "Thickness")
        self.analysis_object.addObject(self.thickness_obj)
        # Load Line Direction
        sh_load_line = Part.makeLine(FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(0, 0, 10))
        load_line = self.doc.addObject("Part::Feature", "Load_direction_line")
        load_line.Shape = sh_load_line
        self.loadDirection = load_line
        # mesh
        self.femmesh_obj = self.doc.addObject('Fem::FemMeshObject', 'RectMesh')
        self.analysis_object.addObject(self.femmesh_obj)

        # constraints displacement
        supportsL, supportsR = self.getsupportReference()
        self.supportL = ObjectsFem.makeConstraintDisplacement(self.doc, "supportsL")
        self.supportL.yDisplacement = 0
        self.supportL.yFree = False
        self.supportL.zDisplacement = 0
        self.supportL.zFree = False
        self.analysis_object.addObject(self.supportL)
        self.supportL.References = supportsL

        self.supportR = ObjectsFem.makeConstraintDisplacement(self.doc, "supportsR")
        self.supportR.xDisplacement = 0
        self.supportR.xFree = False
        self.supportR.yDisplacement = 0
        self.supportR.yFree = False
        self.supportR.zDisplacement = 0
        self.supportR.zFree = False
        self.analysis_object.addObject(self.supportR)
        self.supportR.References = supportsR

        # Loads
        shape_faces = self.part.Shape.Faces
        self.Area = 0
        for i, Face in enumerate(shape_faces):
            LoadFaces = (self.part, "Face" + str(i + 1))
            self.Area += Face.Area
            Load = Face.Area * 0.0006
            force_constraint = ObjectsFem.makeConstraintForce(self.doc, "Force" + str(i + 1))
            force_constraint.References = LoadFaces
            force_constraint.Direction = (self.loadDirection, ["Edge1"])
            force_constraint.Force = str(Load) + " N"
            force_constraint.Reversed = True
            self.analysis_object.addObject(force_constraint)

    def createFEMmesh(self):
        femmesh = Fem.FemMesh()
        node_id = 1
        n = int(np.round(self.width / self.meshsize, 0))
        xList = np.linspace(0, self.width, n)  # in the width direction
        yList = self.pts[:, 1]  # in the length direction
        zlist = self.pts[:, 2]
        for x in xList:
            for i, y in enumerate(yList):
                z = zlist[i]
                femmesh.addNode(x, y, z, node_id)
                node_id += 1

        num_divisions_width = len(xList) - 1
        num_divisions_length = len(yList) - 1
        for i in range(num_divisions_width):
            for j in range(num_divisions_length):
                n1 = i * (num_divisions_length + 1) + j + 1
                n2 = n1 + 1
                n3 = n1 + num_divisions_length + 2
                n4 = n1 + num_divisions_length + 1
                # femmesh.addQuad(n1,n2,n3,n4)
                femmesh.addQuad(n4, n3, n2, n1)
        self.femmesh_obj.FemMesh = femmesh

    def getsupportReference(self):
        shape_edges = self.part.Shape.Edges
        lowest_z = min(np.transpose(self.pts)[-1])
        support_edgesL = []
        support_edgesR = []
        for i, edge in enumerate(shape_edges):
            if (edge.Vertexes[0].X == 0 and edge.Vertexes[1].X == 0 and (
                    math.isclose(edge.Vertexes[0].Z, lowest_z, rel_tol=0.05) or math.isclose(edge.Vertexes[1].Z,
                                                                                             lowest_z, rel_tol=0.05))):
                support_edgesL.append((self.part, "Edge" + str(i + 1)))
            elif (edge.Vertexes[0].X == self.width and edge.Vertexes[1].X == self.width and (
                    math.isclose(edge.Vertexes[0].Z, lowest_z, rel_tol=0.05) or math.isclose(edge.Vertexes[1].Z,
                                                                                             lowest_z, rel_tol=0.05))):
                support_edgesR.append((self.part, "Edge" + str(i + 1)))
        return (support_edgesL, support_edgesR)

    def get_displacement_within_point(self, result_obj, target_x, target_y, radius):
        Nodes_vectors_list = list(result_obj.Mesh.FemMesh.Nodes.values())
        displacements_vectors_list = result_obj.DisplacementVectors
        nodes_array = np.array([(node.x, node.y, node.z) for node in Nodes_vectors_list])
        displacements_array = np.array([(disp.x, disp.y, disp.z) for disp in displacements_vectors_list])
        distances = np.sqrt((nodes_array[:, 0] - target_x) ** 2 + (nodes_array[:, 1] - target_y) ** 2)
        within_radius_indices = np.where(distances <= radius)[0]
        z_displacements = displacements_array[within_radius_indices, 2]
        return max(abs(z_displacements))

    def runAnalysis(self):
        self.fea = ccxtools.FemToolsCcx(solver=self.solver_object)
        self.fea.purge_results()
        self.fea.update_objects()
        self.fea.run()
        self.results = self.doc.getObject('CCX_Results')
        if self.results:
            self.max_disp = self.get_displacement_within_point(self.results, self.width / 2, self.length / 2, 100)
            self.maxStress = np.percentile(self.results.PrincipalMax, 99.5)
            self.volume = (self.Area * self.thickness) / 1000000000
        else:
            self.max_disp = 100000
            self.maxStress = 1000000
            self.volume = 1000000

