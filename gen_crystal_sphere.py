#!/usr/bin/env python3

import os
import sys
import math
import numpy as np
import itertools
import CifFile
import qcelemental as qcel
import time
import shutil
from joblib import Parallel, delayed

# Write an output of xyz coordinate with the molden layout
def write_xyz(output_fname, atom_list):
    """
    Write a coordinate file in the standard xyz format.
    
    output_fname := the name of the xyz file
        E.g. "aspirin.xyz"
    
    atom_list := the list of atoms and their coordinates
    It is a list of tuple.
    E.g.
    [('O', 0.62355, 0.14194, 0.94663),
     ('H', 0.574, 0.031, 0.9354),
     ('H', 0.863, 0.7548, 0.9991)]
    """
    
    with open(output_fname, "w") as of:
        of.write("{}".format(len(atom_list)))
        of.write("\n\n")
        
        for atom in atom_list:
            of.write("{sym:<5} {x:>15.10f} {y:>15.10f} {z:>15.10f}\n".format(sym = atom[0], x = atom[1], y = atom[2], z = atom[3]))


def factors_convert_fract2cartes(cif_data):
    """
    Edge vectors (a, b, c) in fractional coordinate –> (x, y, z) in Cartesian coordinate

    cos(alpha) = b*c/(|b||c|)
    cos(beta) = a*c/(|a||c|)
    cos(gamma) = a*b/(|a||b|)

    a = (a, 0, 0)
    b = (bcos(gamma), bsin(gamma), 0)
    c = (cx, cy, cz)

    x = La*u + Lb*cos(gamma)*v + Lc*cos(beta)*w
    y = Lb*sin(gamma)*v + Lc*((cos(alpha)cos(gamma) - cos(alpha))/sin(gamma))*w
    z = Lc * (sqrt(1 - cos_a**2 - cos_b**2 - cos_g**2 + 2*cos_a*cos_b*cos_g)/sin_g)*w
    """
    
    # Lengths of the unit cell
    La = cif_data["_cell_length_a"]
    Lb = cif_data["_cell_length_b"]
    Lc = cif_data["_cell_length_c"]
    
    # Angles in the unit cell
    alpha = math.radians(cif_data["_cell_angle_alpha"])
    beta = math.radians(cif_data["_cell_angle_beta"])
    gamma = math.radians(cif_data["_cell_angle_gamma"])
    
    cos_a = math.cos(alpha)
    sin_a = math.sin(alpha)

    cos_b = math.cos(beta)
    sin_b = math.sin(beta)

    cos_g = math.cos(gamma)
    sin_g = math.sin(gamma)
    
    ax = La
    # ay = az = 0
    
    bx = Lb * cos_g
    by = Lb * sin_g
    # bz = 0
    
    cx = Lc * cos_b
    cy = Lc * (cos_a - cos_g*cos_b)/sin_g
    cz = Lc * math.sqrt(1 - cos_a**2 - cos_b**2 - cos_g**2 + 2*cos_a*cos_b*cos_g)/sin_g
    
    # Use the volume to check that we calculated the vectors correctly
    V = ax * by * cz
    
    if abs(V - cif_data["_cell_volume"]) > 0.1:
        print("WARNING: Volume calculated with the real vectors is not the same as the volume in CIF file.")
        
    return({"ax": ax, "ay": 0, "az": 0, "bx": bx, "by": by, "bz": 0, "cx": cx, "cy": cy, "cz": cz})


# For an atom
def convert_fract2carte_atom(u, v, w, factors_dict):
    ax = factors_dict["ax"]
    bx = factors_dict["bx"]
    cx = factors_dict["cx"]
    by = factors_dict["by"]
    cy = factors_dict["cy"]
    cz = factors_dict["cz"]
    
    x = ax*u + bx*v + cx*w
    y = by*v + cy*w
    z = cz*w
    
    return(x, y, z)


# For a molecule
def convert_fract2carte_molecule(atoms_list, factors_dict):

    ax = factors_dict["ax"]
    bx = factors_dict["bx"]
    cx = factors_dict["cx"]
    by = factors_dict["by"]
    cy = factors_dict["cy"]
    cz = factors_dict["cz"]
    
    molecule_inCart = []
    
    for atom in atoms_list:
        label, u, v, w = atom
        x = ax*u + bx*v + cx*w
        y = by*v + cy*w
        z = cz*w
        
        molecule_inCart.append((label, x, y, z))
        
    return(molecule_inCart)


# Calculate the threshold for the distance of 2 atom types
# Require qcelement package
def dist_threshold(atom_list, tolerance_value):
    rcov = {}

    for i in atom_list:
        rcov[i] = {}
        for j in atom_list:
            r = qcel.covalentradii.get(i, units="angstrom") + qcel.covalentradii.get(j, units="angstrom") * tolerance_value
            #r = round((covalent_radius_dict[atom_i] + covalent_radius_dict[atom_j]) * tolerance_value, 5)
            rcov[i][j] = r

    return rcov


# Calculate the distance between 2 atoms
# with their xyz coordinates in the cluster
# only using (x, y, z)
def distance(i, j):
    dist = math.sqrt((i[0] - j[0])**2 + (i[1] - j[1])**2 + (i[2] - j[2])**2)
    return dist

# Fragment the big cluster into its individual molecules
def fragment(molecules):
    fragments = []
    while len(molecules) > 0:
        first, *rest = molecules
        first = set(first)

        lf = -1
        while len(first) > lf:
            lf = len(first)

            rest2 = []
            for r in rest:
                if len(first.intersection(set(r)))>0:
                    first |= set(r)
                else:
                    rest2.append(r)     
            rest = rest2

        fragments.append(list(first))
        molecules = rest
        
    return fragments

# Calculate the center of mass of a molecule
def center_of_mass(mol_coord):
    """
    mol_coord is a list of tuples. Each tuple holds an atomic coordinate.
    ('O', -0.8310492693341551, -8.864856732599998, 5.019346296047775)
    xCM = Σmixi/M,  yCM = Σmiyi/M,  zCM = Σmizi/M
    """
    
    total_mass = 0.0
    x_com = 0.0
    y_com = 0.0
    z_com = 0.0
    
    for atom in mol_coord:
        total_mass += qcel.periodictable.to_mass(atom[0]) # sum of the atomic mass
        x_com += atom[1] * qcel.periodictable.to_mass(atom[0])
        y_com += atom[2] * qcel.periodictable.to_mass(atom[0])
        z_com += atom[3] * qcel.periodictable.to_mass(atom[0])
    
    x_com = x_com/total_mass
    y_com = y_com/total_mass
    z_com = z_com/total_mass
    
    return (x_com, y_com, z_com) 


def read_cif(cif_name):
    cif = CifFile.ReadCif(cif_name)
    
    for data in cif:
        cif_dblock = data
        break
    
    cif_data = {}
    
    # Extract CIF data and remove the square brackets in the numbers
    
    
    cif_data["_chemical_name"] = cif_dblock["_chemical_name_systematic"]
    if cif_dblock["_chemical_name_systematic"] == "?":
        cif_data["_chemical_name"] = cif_dblock["_chemical_name_common"]
    
    cif_data["_chemical_formula_moiety"] = cif_dblock["_chemical_formula_moiety"]
    cif_data["_cell_length_a"] = float(cif_dblock["_cell_length_a"].replace("(", "").replace(")", ""))
    cif_data["_cell_length_b"] = float(cif_dblock["_cell_length_b"].replace("(", "").replace(")", ""))
    cif_data["_cell_length_c"] = float(cif_dblock["_cell_length_c"].replace("(", "").replace(")", ""))
    cif_data["_cell_angle_alpha"] = float(cif_dblock["_cell_angle_alpha"].replace("(", "").replace(")", ""))
    cif_data["_cell_angle_beta"] = float(cif_dblock["_cell_angle_beta"].replace("(", "").replace(")", ""))
    cif_data["_cell_angle_gamma"] = float(cif_dblock["_cell_angle_gamma"].replace("(", "").replace(")", ""))
    cif_data["_cell_volume"] = float(cif_dblock["_cell_volume"].replace("(", "").replace(")", ""))

    # Extract the symmetry operations that define the space group
    '''
    In some cases, it might be called "_space_group_symop_operation_xyz".  
    In the CIF file, the symmetry-equivalent position in the xyz format look like: 
    ```
    loop_  
    _symmetry_equiv_pos_as_xyz  
        'x,y,z'  
        'y,x,2/3-z'  
        '-y,x-y,2/3+z'  
        '-x,-x+y,1/3-z'  
        '-x+y,-x,1/3+z'  
        'x-y,-y,-z'  
    ```
    Except for the space group P1, these data will be repeated in a loop.
    '''
    cif_data["_symmetry_equiv_pos_as_xyz"] = []
    
    try:
        sym_op = cif_dblock["_symmetry_equiv_pos_as_xyz"]
    except KeyError:
        try:
            sym_op = cif_dblock["_space_group_symop_operation_xyz"]
        except KeyError:
            print("\n ERROR: Cif file does not have an item: either \"_symmetry_equiv_pos_as_xyz\" or \"_space_group_symop_operation_xyz\".")
            sys.exit()
    
    for xyz_op in sym_op:
        cif_data["_symmetry_equiv_pos_as_xyz"].append(xyz_op)
    
    # Get the fractional coordinates u, v, w (x, y, z) of the atoms
    cif_data["_atom_site_label"] = cif_dblock["_atom_site_label"]
    cif_data["_atom_site_type_symbol"] = cif_dblock["_atom_site_type_symbol"]
    
    cif_data["_atom_site_fract_x"] = []
    for u in cif_dblock["_atom_site_fract_x"]:
        cif_data["_atom_site_fract_x"].append(float(u.split("(")[0]))
        
    cif_data["_atom_site_fract_y"] = []
    for v in cif_dblock["_atom_site_fract_y"]:
        cif_data["_atom_site_fract_y"].append(float(v.split("(")[0]))
        
    cif_data["_atom_site_fract_z"] = []
    for w in cif_dblock["_atom_site_fract_z"]:
        cif_data["_atom_site_fract_z"].append(float(w.split("(")[0]))

    return(cif_data)


def asym_unit(cif_data):
    
    # Atom labels
    atom_labels = cif_data["_atom_site_type_symbol"]
    # Atom coordinates
    atom_u = cif_data["_atom_site_fract_x"]
    atom_v = cif_data["_atom_site_fract_y"]
    atom_w = cif_data["_atom_site_fract_z"]
    
    asym_unit = []
    asym_unit = [(atom_labels[i], atom_u[i], atom_v[i], atom_w[i]) for i in range(len(atom_labels))]
    
    # Move atoms into a unit cell
    asym_unit = [(atom[0], atom[1]%1.0, atom[2]%1.0, atom[3]%1.0) for atom in asym_unit]
    
    return(asym_unit)


def unit_cell(atoms, cif_data):
    '''
    Use symmetry operations to create the unit cell

    The CIF file consists of a few atom positions and several "symmetry operations" that indicate the other atom positions within the unit cell.  
    Using these symmetry operations, create copies of the atoms until no new copies can be made.  
    
    For each atom, apply each symmetry operation to create a new atom.
    '''
    
    # Symmetry operation
    sym_op = cif_data["_symmetry_equiv_pos_as_xyz"]
    
    imax = len(atoms)
    i = 0

    atoms_uc = []

    while i < imax:
        label, x, y, z = atoms[i] 
        # Keep x, y, z as they are! Cause they will be inserted in the eval(op) later.

        for op in sym_op:
            # eval will convert the string into a 3-tuple using the current values for x, y, z
            u, v, w = eval(op)

            # Move new atom into the unit cell
            u = u % 1.0
            v = v % 1.0
            w = w % 1.0

            # Check if the new position is actually new, or already exists
            # Two atoms are on top of each other if they are less than "eps" away.
            eps = 0.01

            new_atom = True

            for atom in atoms:
                if (abs(atom[1] - u) < eps) and (abs(atom[2] - v) < eps) and (abs(atom[3] - w) < eps):
                    new_atom = False

                    # Check that this is the same atom type.
                    if atom[0] != label:
                        print("\nERROR: Invalid CIF file: atom of type %s overlaps with atom of type %s" % (atom[0], label))

            if (new_atom):
                atoms.append((label, u, v, w))

        i = i + 1
        imax = len(atoms)
    
    atoms_uc = atoms
    
    return(atoms_uc)


def supercell(atoms_uc, Nx, Ny, Nz):
    atoms_sc = []

    for atom in atoms_uc:

        label, u, v, w = atom

        for i in range(Nx):
            uu = i + u

            for j in range(Ny):
                vv = j + v

                for k in range(Nz):
                    ww = k + w
                    atoms_sc.append((label, uu, vv, ww))
    
    return(atoms_sc)


def convert_supercell_tocartes(atoms_sc, cif_data, Nx, Ny, Nz):
    factors_fract2carte_dict = factors_convert_fract2cartes(cif_data)
    
    # Check if we have a rectangular box

    rect_box = False

    bx = factors_fract2carte_dict["bx"]
    cx = factors_fract2carte_dict["cx"]
    cy = factors_fract2carte_dict["cy"]

    eps = 0.1

    if (bx < eps) and (cx < eps) and (cy < eps):
        rect_box = True
        
    # Calculate the box size
    Lx = Nx * cif_data['_cell_length_a']
    Ly = Ny * cif_data['_cell_length_b']
    Lz = Nz * cif_data['_cell_length_c']
    
    atoms_rsc = []

    for atom in atoms_sc:
        label, xf, yf, zf = atom
        (xn, yn, zn) = convert_fract2carte_atom(xf, yf, zf, factors_fract2carte_dict)

        if rect_box:
            xn = (xn + Lx) % Lx
            yn = (yn + Ly) % Ly
            zn = (zn + Lz) % Lz

        atoms_rsc.append((label, xn, yn, zn))
    
    return(atoms_rsc)


def finalise_supercell(atoms_rsc):
    '''
    Clean the duplicates in the supercell coordinates
    Translate the supercell to the origin
    '''
    
    # Make sure there is no duplicate rows in the super cell coordinates
    clean_sc, uniq_idx = np.unique(atoms_rsc, return_index=True, axis=0)
    
    sc_coord = []
    sc_elem = []

    for atom in clean_sc:
        sc_coord.append([float(atom[1]), float(atom[2]), float(atom[3])])
        sc_elem.append(atom[0])

    sc_coord = np.array(sc_coord)
    sc_elem = np.array(sc_elem)

    # Find the origin of the super cell
    center_sc = (np.max(sc_coord, axis=0) - np.min(sc_coord, axis=0))/2

    # Translate the supercell to the origin
    sc_coord -= center_sc
    
    # Glue the coordinate and elements together
    sc_rcoord = [(sc_elem[i], sc_coord[i][0], sc_coord[i][1], sc_coord[i][2]) for i in range(len(sc_elem))]
    
    return(sc_rcoord)

# Don't need this function anymore
def gen_cov_sparse_mat_old(sc_rcoord):
    
    print(".. Start with decomposing the supercell.")
    uniq_atoms = list(set([i[0] for i in sc_rcoord]))
    rcov_dict = dist_threshold(uniq_atoms, 1.2)
    
    sparse_mat = np.zeros(shape=(len(sc_rcoord), len(sc_rcoord)), dtype=int)
    
    start_time = time.time()
    
    for i in range(len(sc_rcoord)):
        for j in range(i,len(sc_rcoord)):  
            if distance(sc_rcoord[i][1:],sc_rcoord[j][1:]) < rcov_dict[sc_rcoord[i][0]][sc_rcoord[j][0]]:
                sparse_mat[i,j] = 1
    
    print(".. {} seconds".format(round(time.time() - start_time, 1)))
    print(".. Done with writing the sparse matrix for covalent bonding.")
    
    return(sparse_mat)

# Don't need this function anymore
def gen_cov_sparse_mat(sc_rcoord):

    max_covalency = {"H": 1, "C": 4, "N": 3, "O": 2}
    
    uniq_atoms = list(set([i[0] for i in sc_rcoord]))
    rcov_dict = dist_threshold(uniq_atoms, 1.2)
    
    sparse_mat = np.zeros(shape=(len(sc_rcoord), len(sc_rcoord)), dtype=int)
    d = distance
    
    print(".. Start with making the covalent sparse matrix.")
    start_time = time.time()
    
    for i in range(len(sc_rcoord)):
        count_cvb = 0
        max_cvb = max_covalency[sc_rcoord[i][0]]
        
        for j in range(i,len(sc_rcoord)):  
            if d(sc_rcoord[i][1:],sc_rcoord[j][1:]) < rcov_dict[sc_rcoord[i][0]][sc_rcoord[j][0]]:
                sparse_mat[i,j] = 1
                count_cvb += 1
                
            if count_cvb == max_cvb + 1: break
    
    print(".. {} seconds".format(round(time.time() - start_time, 1)))
    print(".. Done with writing the sparse matrix for covalent bonding.")
    
    return(sparse_mat)

# Don't need this function anymore
def decompose_supercell(sc_rcoord):
    
    sparse_mat = gen_cov_sparse_mat(sc_rcoord)
    
    start_time = time.time()
    molecules = []

    for i in range(len(sc_rcoord)):
        atom_connect_list = []
        atom_connect_list.append(i)

        for j in range(i,len(sc_rcoord)):
            if sparse_mat[i,j] == 1:
                atom_connect_list.append(j)

        #atom_connect_list.sort()
        molecules.append(atom_connect_list)
    
    print(".. {} seconds".format(round(time.time() - start_time, 1)))
    print(".. Done with making the list of connecting atoms")
    
    start_time = time.time()
    fragments = fragment(molecules)
    print(".. {} seconds".format(round(time.time() - start_time, 1)))
    
    return(fragments)


# Don't need this function anymore
def truncate_fragments(fragments, nr_atoms):
    '''
    Compare the size of fragment with the number of atoms in a molecule
    len(cif_data["_atom_site_label"]) holds the atoms in a molecule
    '''
    
    unbroken_fragments = []
    
    for i in fragments:
    
        if len(i) == nr_atoms: 
            unbroken_fragments.append(i)
            
    return(unbroken_fragments)


def extract_xyz_fragments(unbroken_fragments, sc_rcoord):
    xyz_fragments = {}
    
    for i in range(0,len(unbroken_fragments)):
        xyz_fragments[i+1] = [sc_rcoord[atom] for atom in unbroken_fragments[i]]
        
    return(xyz_fragments)


def calc_center_of_mass(xyz_fragments):
    '''
    Output: a dictionary
    '''
    com_fragments = {}

    for idx,mol in xyz_fragments.items():
        com_fragments[idx] = center_of_mass(mol)
        
    return(com_fragments)


def find_refmol(com_fragments):
    
    # convert the coordinates of COM stored in the dictionary into the numpy array
    com_coord_array = np.array(list(com_fragments.values()))
    
    # find the median for x, y, z coordinates of COM
    x_med = np.median(com_coord_array[:,0])
    y_med = np.median(com_coord_array[:,1])
    z_med = np.median(com_coord_array[:,2])
    ref_point = (x_med, y_med, z_med) 

    
    # find the reference molecule that is closest to the median point
    # pair is the index of the molecule and its distance to the reference point which is the smallest
    pair = (0, 1000.0)

    for key,val in com_fragments.items():
        dist2ref = distance(val, ref_point)
        if dist2ref <= pair[1]:
            pair = (key, dist2ref)  

    idx_refmol = pair[0]
    
    return(idx_refmol)


def calc_dCOMs_other2ref(com_fragments, idx_refmol):
    
    dCOMs_other2ref = {}
    
    for key, val in com_fragments.items():
        r = round(distance(com_fragments[idx_refmol], val), 1)
        dCOMs_other2ref[key] = r
    
    return(dCOMs_other2ref)


def gen_cryst_sphere(r_thres, idx_refmol, dCOMs_other2ref, xyz_fragments):
    
    cryst_sphere = {}
    dCOMs_sphere = {}
    
    count = 1
    
    for key,val in dCOMs_other2ref.items():
        if val <= r_thres:
            cryst_sphere[count] = xyz_fragments[key]
            dCOMs_sphere[count] = val
            
            if key == idx_refmol:
                new_idx_refmol = count
            
            count += 1
    
    return(cryst_sphere, dCOMs_sphere, new_idx_refmol)


def xyz_dict2list(mol_dict):
    xyz_list = []
    
    for mol in mol_dict.values():
        for atom in mol:
            xyz_list.append(atom)
            
    return(xyz_list)


def mol_in_dimers(r_dim_thres, dCOMs_sphere):
    idx_dimers = []
    
    for key,val in dCOMs_sphere.items(): 
        if val <= r_dim_thres:
            idx_dimers.append(key)
            
    idx_dimers.sort()
            
    return(idx_dimers)



def gen_dimer_jobs(idx_dimers, idx_refmol, cryst_sphere):
    maindir = os.getcwd()
    try:
        os.mkdir("dimers")
    except FileExistsError:
        shutil.rmtree("dimers")
        os.mkdir("dimers")

    os.chdir("dimers/")

    ghost_refmol = [("@" + atom[0], atom[1], atom[2], atom[3]) for atom in cryst_sphere[idx_refmol]]

    for i in idx_dimers:
        if i != idx_refmol:
            dimer = [idx_refmol, i]
            dimer.sort()
            dim_name = "_".join(["dimer", str(dimer[0]), str(dimer[1])])
            dirname = dim_name + "/"
            os.mkdir(dirname)
            
            # write dimer xyz file
            write_xyz(atom_list = cryst_sphere[dimer[0]] + cryst_sphere[dimer[1]], output_fname = dirname + dim_name + ".xyz")

            # write NoCP reference molecule
            write_xyz(atom_list = cryst_sphere[idx_refmol], output_fname = dirname + "ncp_mono_" + str(idx_refmol) + ".xyz")
            # write NoCP other molecule
            write_xyz(atom_list = cryst_sphere[i], output_fname = dirname + "ncp_mono_" + str(i) + ".xyz")

            ghost_mol_i = [("@" + atom[0], atom[1], atom[2], atom[3]) for atom in cryst_sphere[i]]
            # write CP reference molecule
            write_xyz(atom_list = cryst_sphere[idx_refmol] + ghost_mol_i, output_fname = dirname + "cp_mono_" + str(idx_refmol) + ".xyz")
            # write CP other molecule
            write_xyz(atom_list = cryst_sphere[i] + ghost_refmol, output_fname = dirname + "cp_mono_" + str(i) + ".xyz")

    os.chdir(maindir)   
    
    
def mol_in_trimers(r_trim_thres, dCOMs_sphere):
    idx_trimers = []
    
    for key,val in dCOMs_sphere.items(): 
        if val <= r_trim_thres:
            idx_trimers.append(key)
    
    idx_trimers.sort()
    
    return(idx_trimers)



def gen_trimer_jobs(idx_trimers, idx_refmol, cryst_sphere):
    maindir = os.getcwd()

    try:
        os.mkdir("trimers")
    except FileExistsError:
        shutil.rmtree("trimers")
        os.mkdir("trimers")

    os.chdir("trimers/")

    ghost_refmol = [("@" + atom[0], atom[1], atom[2], atom[3]) for atom in cryst_sphere[idx_refmol]]

    for i in idx_trimers:
        for j in idx_trimers[(idx_trimers.index(i) + 1):]: 
            if i != idx_refmol and j != idx_refmol:
                trimer = [idx_refmol, i, j]
                trimer.sort()
                trim_name = "_".join(["trimer", str(trimer[0]), str(trimer[1]), str(trimer[2])])

                dirname = trim_name + "/"
                os.mkdir(dirname)

                # write trimer xyz file
                write_xyz(atom_list = cryst_sphere[trimer[0]] + cryst_sphere[trimer[1]] + cryst_sphere[trimer[2]], output_fname = dirname + trim_name + ".xyz")

                ghost_mol_dict = {}
                for mol in trimer:
                    ghost_mol_dict[mol] = [("@" + atom[0], atom[1], atom[2], atom[3]) for atom in cryst_sphere[mol]]

                # Monomers
                for mol in trimer:
                    # write NoCP xyz files for monomer
                    write_xyz(atom_list = cryst_sphere[mol], output_fname = dirname + "ncp_mono_" + str(mol) + ".xyz")

                    # write CP xyz files for monomer
                    tmp = trimer.copy()
                    tmp.remove(mol)
                    write_xyz(atom_list = cryst_sphere[mol] + ghost_mol_dict[tmp[0]] + ghost_mol_dict[tmp[1]], output_fname = dirname + "cp_mono_" + str(mol) + ".xyz")

                # Dimers
                for mol_1 in trimer:
                    for mol_2 in trimer[(trimer.index(mol_1) + 1):]:
                         # write NoCP xyz file for dimers
                        write_xyz(atom_list = cryst_sphere[mol_1] + cryst_sphere[mol_2], output_fname = dirname + "ncp_dimer_" + str(mol_1) + "_" + str(mol_2) + ".xyz")

                        # write CP xyz file for dimer
                        tmp = trimer.copy()
                        tmp.remove(mol_1)
                        tmp.remove(mol_2)

                        write_xyz(atom_list = cryst_sphere[mol_1] + cryst_sphere[mol_2] + ghost_mol_dict[tmp[0]], output_fname = dirname + "cp_dimer_" + str(mol_1) + "_" + str(mol_2) + ".xyz")      

    os.chdir(maindir)
    

def ask_for_threshold(cif_data):
    cell_vol = cif_data["_cell_volume"]
    a = cell_vol**(1./3.)
    
    given_r = input("Do you want to give specific radius values? (Y/N) ")
    print("\n")
    
    r_thres = r_dim_thres = r_trim_thres = 0.0
    Nx = Ny = Nz = 1
    
    if given_r == "Y" or given_r == "y":
        r_thres = float(input("The radius value for crystalline sphere (in Angstrom): "))
        
        while a > r_thres:
            print('''\nWARNING:\nThe radius value for the crystalline sphere is too small.\nIt should be at least {}.\n'''.format(a*3))
            r_thres = float(input("Choose another value for radius: "))
            
        r_dim_thres = float(input("The radius value for dimers (in Angstrom): "))
        r_trim_thres = float(input("The radius value for trimers (in Angstrom): "))
        
        Nx = Ny = Nz = 2 * int(r_thres/a + 2)
        
    elif given_r == "N" or given_r == "n":
        a = cell_vol**(1./3.)   # volume = a^3
        r_thres = round(a*3, 1)   # R = a*3
        r_dim_thres = round(r_thres/2.0, 1)
        r_trim_thres = round(r_thres/3.0 + 1, 1)
        
        Nx = Ny = Nz = 8  
        # R = a*3 => at least 3 unit cells 
        # => add one more to ensure a full sphere after truncation
        # in both directions: 4*2 = 8
           
    else:
        print("\n ERROR: You should answer with Y/y or N/n. \n")
        sys.exit()
    
    print("\n*******************************************************")
    print("SUMMARY OF INPUT VALUES\n")
    print("The radius of the crystalline sphere:  {}  Angstrom".format(r_thres))
    print("The threshold for dimers:              {}  Angstrom".format(r_dim_thres))
    print("The threshold for trimers:             {}  Angstrom".format(r_trim_thres))
    print("The size of supercell:                 {} x {} x {}".format(Nx, Ny, Nz))
    print("********************************************************\n")
    
    return(r_thres, r_dim_thres, r_trim_thres, Nx, Ny, Nz)

    
def main_run_as_script():
    
    # Step 1: 
    ## Prompt questions to the user
    cif_file = input("The CIF file name: ")
    fpath_cif = os.path.abspath(cif_file)
    
    ## Make a new directory to save output files
    casename = cif_file.split(".")[0]
    maindir = os.getcwd()
    
    try:
        os.mkdir(casename)
    except FileExistsError:
        shutil.rmtree(casename)
        os.mkdir(casename)
    
    shutil.copy(fpath_cif, casename) # copy the cif file into the folder 
        
    os.chdir(casename)
    
    # Step 2: 
    ## Read into the CIF file and extract data from it into a dictionary
    cif_data = read_cif(cif_file)
    print("\n***\nStart the engine for {}, {}\n***\n".format(cif_data["_chemical_name"], cif_data["_chemical_formula_moiety"]))
    
    
    # Step 3:
    ## Ask for other threshold values
    r_thres, r_dim_thres, r_trim_thres, Nx, Ny, Nz = ask_for_threshold(cif_data)
    
    # Step 4:
    ## Get the asymmetric unit from CIF dictionary
    atoms = asym_unit(cif_data)
    print(".. Done with extracting the asymmetric unit.")
    
    # Step 5:
    ## Create the unit cell with symmetry operations
    atoms_uc = unit_cell(atoms, cif_data)
    print(".. Done with creating the unit cell.")
    
    # Step 6:
    ## Create the supercell 
    atoms_sc = supercell(atoms_uc, Nx, Ny, Nz)
    
    # Step 7:
    ## Convert the fractional coordinates into the real coordinates of the supercell
    atoms_rsc = convert_supercell_tocartes(atoms_sc, cif_data, Nx, Ny, Nz)
    
    # Step 8:
    ## Clean and translate the supercell
    sc_rcoord = finalise_supercell(atoms_rsc)
    print(".. Done with creating the supercell.")
    print("There are {} atoms in the supercell.".format(len(sc_rcoord)))
    
    # Step 9:
    ## Fragment the supercell into individual monomers
    fragments = decompose_supercell(sc_rcoord)
    print(".. Done with fragmenting the supercell.")
    
    unbroken_fragments = truncate_fragments(fragments, len(cif_data["_atom_site_label"]))
    print(".. Done with filtrating the broken fragments.")
    
    xyz_fragments = extract_xyz_fragments(unbroken_fragments, sc_rcoord)
    print(".. Done with obtaining xyz coordinates for the fragments.")
    
    # Step 10:
    ## Calculate the center of mass for each monomers
    com_fragments = calc_center_of_mass(xyz_fragments)
    print(".. Done with calculating the centers of mass for each fragment.")
    
    # Step 11:
    ## Find the reference molecule
    idx_refmol = find_refmol(com_fragments)
    print("\nThe index of reference molecule is {}.\n".format(idx_refmol))
    print(".. Done with finding the reference molecule.")
    
    ## Calculate the distance of COMs
    dCOMs_other2ref = calc_dCOMs_other2ref(com_fragments, idx_refmol)
    print(".. Done with calculating distance of COMs between the reference molecule and others.")
    
    # Step 12:
    ## Generate the crystalline sphere around the reference molecule
    cryst_sphere = gen_cryst_sphere(r_thres, dCOMs_other2ref, xyz_fragments)
    print("\nThere are {} molecules in the crystalline sphere.".format(len(cryst_sphere.keys())))
          
    xyz_cryst_sphere = xyz_dict2list(cryst_sphere)
    print("There are {} atoms in the crystalline sphere.\n".format(len(xyz_cryst_sphere)))
    write_xyz(atom_list = xyz_cryst_sphere, output_fname = "{}_sphere.xyz".format(casename))
    print(".. Done with generating the crystalline sphere.")
    
    # Step 13:
    ## Generate the dimer jobs
    idx_dimers = mol_in_dimers(r_dim_thres, dCOMs_other2ref)
    gen_dimer_jobs(idx_dimers, idx_refmol, cryst_sphere)
    print(".. Done with generating the dimer jobs.")
    
    # Step 14:
    ## Generate the trimer jobs
    idx_trimers = mol_in_trimers(r_trim_thres, dCOMs_other2ref)
    gen_trimer_jobs(idx_trimers, idx_refmol, cryst_sphere)
    print(".. Done with generating the trimer jobs.")
    
    print("\nSucceeded in generating jobs for your lattice energy calculation!\n")
    
    os.chdir(maindir)
    sys.exit()


def read_input(inputfile):
    with open(inputfile, "r") as f:
        inp = f.read().splitlines()
        inp = [i for i in inp if i[0:2] != "//"]
        inp_dict = {i.split(":")[0].strip() : i.split(":")[1].strip() for i in inp}
        
    return(inp_dict)


def find_cif(inp_dict):
    try:
        cif_file = inp_dict["Cif"]
    except KeyError:
        print("ERROR: cannot find 'Cif' in the input file.")
    
    return(cif_file)


def check_given_thresholds(inp_dict):
    try:
        given_r = inp_dict["Rgiven"]
    except KeyError:
        print("ERROR: cannot find 'Rgiven' in the input file.")

    if given_r == "y" or given_r == "Y": return(True)    
    elif given_r == "n" or given_r == "N": return(False)
    else:
        print("\n ERROR: You should answer with Y/y or N/n. \n")
        sys.exit()
        

def read_threshold_values(inp_dict, cif_data):

    try:
        r_thres = float(inp_dict["Rsphere"])
    except KeyError:
        print("ERROR: cannot find 'Rsphere' in the input file.")
        
    cell_vol = cif_data["_cell_volume"]
    cell_length = cell_vol**(1./3.)

    if cell_length > r_thres:
        print("\nWARNING:\nThe radius value for the crystalline sphere is too small.\nIt should be at least {}.".format(round(cell_length*3, 1)))
        print("The process is keep going on.\n")
        
    try: 
        r_dim_thres = float(inp_dict["Rdim"])
    except KeyError:
        print("ERROR: cannot fine 'Rdim' in the input file.")

    try:
        r_trim_thres = float(inp_dict["Rtrim"])
    except KeyError:
        print("ERROR: cannot find 'Rtrim' in the input file.")
    

        
    Nx = math.ceil(r_thres/cif_data["_cell_length_a"] + 1) * 2
    Ny = math.ceil(r_thres/cif_data["_cell_length_b"] + 1) * 2
    Nz = math.ceil(r_thres/cif_data["_cell_length_c"] + 1) * 2       
    
    print("\n*******************************************************")
    print("SUMMARY OF INPUT VALUES\n")
    print("The radius of the crystalline sphere:  {}  Angstrom".format(r_thres))
    print("The threshold for dimers:              {}  Angstrom".format(r_dim_thres))
    print("The threshold for trimers:             {}  Angstrom".format(r_trim_thres))
    print("The size of supercell:                 {} x {} x {}".format(Nx, Ny, Nz))
    print("********************************************************\n")
    
    return(r_thres, r_dim_thres, r_trim_thres, Nx, Ny, Nz)


def calculate_threshold_values(cif_data):
    
    cell_vol = cif_data["_cell_volume"]
    cell_length = cell_vol**(1./3.)
    
    r_thres = round(cell_length*2, 1)   # R = a*3
    r_dim_thres = round(r_thres/2.0, 1)
    r_trim_thres = round(r_thres/3.0 + 1, 1)

    # a - Nx, b - Ny, c - Nz
    Nx = math.ceil(r_thres/cif_data["_cell_length_a"] + 1) * 2
    Ny = math.ceil(r_thres/cif_data["_cell_length_b"] + 1) * 2
    Nz = math.ceil(r_thres/cif_data["_cell_length_c"] + 1) * 2           
    
    print("\n*******************************************************")
    print("SUMMARY OF INPUT VALUES\n")
    print("The radius of the crystalline sphere:  {}  Angstrom".format(r_thres))
    print("The threshold for dimers:              {}  Angstrom".format(r_dim_thres))
    print("The threshold for trimers:             {}  Angstrom".format(r_trim_thres))
    print("The size of supercell:                 {} x {} x {}".format(Nx, Ny, Nz))
    print("********************************************************\n")
    
    return(r_thres, r_dim_thres, r_trim_thres, Nx, Ny, Nz)
    

def main_1(inputfile):

    start_time = time.time()   
    
    # Step 1: 
    ## Read into the input file from the command line
    inp_dict = read_input(inputfile)
    
    ## Check if the cif file name can be found
    cif_file = find_cif(inp_dict)
    casename = cif_file.split(".")[0]
    
    # Step 2: 
    ## Read into the CIF file and extract data from it into a dictionary
    cif_data = read_cif(cif_file)
    print("\n***\nStart the engine for {}, {}\n***\n".format(cif_data["_chemical_name"], cif_data["_chemical_formula_moiety"]))
    
    # Step 3:
    ## Get threshold values
    given_r = check_given_thresholds(inp_dict)
    
    if given_r:
        r_thres, r_dim_thres, r_trim_thres, Nx, Ny, Nz = read_threshold_values(inp_dict, cif_data)
        
    else:
        r_thres, r_dim_thres, r_trim_thres, Nx, Ny, Nz = calculate_threshold_values(cif_data)
    
    # Step 4:
    ## Get the asymmetric unit from CIF dictionary
    atoms = asym_unit(cif_data)
    print(".. Done with extracting the asymmetric unit.")
    
    # Step 5:
    ## Create the unit cell with symmetry operations
    atoms_uc = unit_cell(atoms, cif_data)
    print(".. Done with creating the unit cell.")
    
    # Step 6:
    ## Create the supercell 
    atoms_sc = supercell(atoms_uc, Nx, Ny, Nz)
    
    # Step 7:
    ## Convert the fractional coordinates into the real coordinates of the supercell
    atoms_rsc = convert_supercell_tocartes(atoms_sc, cif_data, Nx, Ny, Nz)
    
    # Step 8:
    ## Clean and translate the supercell
    sc_rcoord = finalise_supercell(atoms_rsc)
    print(".. Done with creating the supercell.")
    print("There are {} atoms in the supercell.".format(len(sc_rcoord)))
    
    # Step 9:
    ## Fragment the supercell into individual monomers
    fragments = decompose_supercell(sc_rcoord)
    print(".. Done with fragmenting the supercell.")
    
    ## !!! Remove this code for truncating the residues in the supercell
    ## In some CIF file, the molecules might have symmetry, so athat only a part of the molecule is recorded. This leads to false fragmenting.
    #unbroken_fragments = truncate_fragments(fragments, len(cif_data["_atom_site_label"]))
    #print(".. Done with filtrating the broken fragments.")
    
    # read into all fragments: molecules + residues
    xyz_fragments = extract_xyz_fragments(fragments, sc_rcoord)
    print(".. Done with obtaining xyz coordinates for the fragments.")
    
    print(".. Write xyz of all fragments into a text file.")
    with open("xyz_fragments_dict.txt", "w") as f:
        f.write(str(xyz_fragments))

    # Step 10:
    ## Calculate the center of mass for each monomers
    com_fragments = calc_center_of_mass(xyz_fragments)
    print(".. Done with calculating the centers of mass for each fragment.")
    
    # Step 11:
    ## Find the index of reference molecule in the supercell
    idx_refmol = find_refmol(com_fragments)
    print(".. Done with finding the reference molecule.")
    
    ## Calculate the distance of COMs between the reference molecule and other fragments
    dCOMs_other2ref = calc_dCOMs_other2ref(com_fragments, idx_refmol)
    print(".. Done with calculating distance of COMs between the reference molecule and others.")
    
    # Step 12:
    ## Generate the crystalline sphere around the reference molecule
    ## Make a new list of COM distance in there sphere
    ## Find the new index for the reference molecule
    (cryst_sphere, dCOMs_sphere, idx_refmol) = gen_cryst_sphere(r_thres, idx_refmol, dCOMs_other2ref, xyz_fragments)
    print("\nThere are {} molecules in the crystalline sphere.".format(len(cryst_sphere.keys())))
    print("\nThe index of reference molecule is {}.\n".format(idx_refmol))
    
    ## Check if all fragments in the crystalline sphere have the same number of atoms
    cryst_sphere_frags = list(cryst_sphere.values())
    isSimilar = all(len(element) == len(cryst_sphere_frags[0]) for element in cryst_sphere_frags)
    
    if (isSimilar): print("All molecules in the sphere are the same.")
    else:
        print("There are still residues existing in the sphere.")
        print("Write the dictionary of xyz molecules to file cryst_sphere_dict.txt")
        
        with open("cryst_sphere_dict.txt", "w") as f:
            f.write(str(cryst_sphere))
        
        print("\nEnd process with issue!\n")
        sys.exit()
          
    xyz_cryst_sphere = xyz_dict2list(cryst_sphere)
    print("There are {} atoms in the crystalline sphere.\n".format(len(xyz_cryst_sphere)))
    write_xyz(atom_list = xyz_cryst_sphere, output_fname = "{}_sphere.xyz".format(casename))

    print(".. Done with generating the crystalline sphere.")
    
    # Step 13:
    ## Generate the dimer jobs
    idx_dimers = mol_in_dimers(r_dim_thres, dCOMs_sphere)
    ndimers = len(idx_dimers) - 1
    print("Number of dimers: {}".format(ndimers))
    gen_dimer_jobs(idx_dimers, idx_refmol, cryst_sphere)
    print(".. Done with generating the dimer jobs.")
    
    # Step 14:
    ## Generate the trimer jobs
    idx_trimers = mol_in_trimers(r_trim_thres, dCOMs_sphere)
    ntrimers = int((len(idx_trimers) - 1)*(len(idx_trimers) - 2)/2)
    print("Number of trimers: {}".format(ntrimers))
    gen_trimer_jobs(idx_trimers, idx_refmol, cryst_sphere)
    print(".. Done with generating the trimer jobs.")
    
    print("\nSucceeded in generating jobs for your lattice energy calculation!")
    
    end_time = time.time()
    print("The process took {} minute(s)\n".format(round((end_time - start_time)/60.0, 2)))
    
    sys.exit()
    

def process_molecules(i, sc_rcoord, rcov_dict):
    start_time = time.time()

    d = distance
    l = [j for j in range(i, len(sc_rcoord)) if d(sc_rcoord[i][1:],sc_rcoord[j][1:]) < rcov_dict[sc_rcoord[i][0]][sc_rcoord[j][0]]]
    return l

def make_molecule(sc_rcoord, rcov_dict):
    start_time = time.time()
    molecules = []
    
    for i in range(len(sc_rcoord)):
        l = [j for j in range(i, len(sc_rcoord)) if distance(sc_rcoord[i][1:],sc_rcoord[j][1:]) < rcov_dict[sc_rcoord[i][0]][sc_rcoord[j][0]]]
        molecules.append(l)
        
    print(".. {} seconds".format(round(time.time() - start_time, 1)))
    print(".. Done with making the list of connecting atoms")
    
    return(molecules)
    


def main(inputfile):

    start_time = time.time()   
    
    # Step 1: 
    ## Read into the input file from the command line
    inp_dict = read_input(inputfile)
    
    ## Check if the cif file name can be found
    cif_file = find_cif(inp_dict)
    casename = cif_file.split(".")[0]
    
    # Step 2: 
    ## Read into the CIF file and extract data from it into a dictionary
    cif_data = read_cif(cif_file)
    print("\n***\nStart the engine for {}, {}\n***\n".format(cif_data["_chemical_name"], cif_data["_chemical_formula_moiety"]))
    
    # Step 3:
    ## Get threshold values
    given_r = check_given_thresholds(inp_dict)

    if given_r:
        r_thres, r_dim_thres, r_trim_thres, Nx, Ny, Nz = read_threshold_values(inp_dict, cif_data)
        
    else:
        r_thres, r_dim_thres, r_trim_thres, Nx, Ny, Nz = calculate_threshold_values(cif_data)
    
    # Step 4:
    ## Get the asymmetric unit from CIF dictionary
    atoms = asym_unit(cif_data)
    print(".. Done with extracting the asymmetric unit.")
    
    # Step 5:
    ## Create the unit cell with symmetry operations
    atoms_uc = unit_cell(atoms, cif_data)
    print(".. Done with creating the unit cell.")
    
    # Step 6:
    ## Create the supercell 
    atoms_sc = supercell(atoms_uc, Nx, Ny, Nz)
    
    # Step 7:
    ## Convert the fractional coordinates into the real coordinates of the supercell
    atoms_rsc = convert_supercell_tocartes(atoms_sc, cif_data, Nx, Ny, Nz)
    
    # Step 8:
    ## Clean and translate the supercell
    sc_rcoord = finalise_supercell(atoms_rsc)
    print(".. Done with creating the supercell.")
    print("There are {} atoms in the supercell.".format(len(sc_rcoord)))
    
    # Step 9:
    ## Fragment the supercell into individual monomers
    uniq_atoms = list(set([i[0] for i in sc_rcoord]))
    rcov_dict = dist_threshold(uniq_atoms, 1.2)
    #molecules = Parallel(n_jobs=4)(delayed(process_molecules)(i, sc_rcoord, rcov_dict) for i in range(len(sc_rcoord)))
    molecules = make_molecule(sc_rcoord, rcov_dict)
    
    fragments = fragment(molecules)
    print(".. Done with fragmenting the supercell.")
    
    ## !!! Remove this code for truncating the residues in the supercell
    ## In some CIF file, the molecules might have symmetry, so athat only a part of the molecule is recorded. This leads to false fragmenting.
    #unbroken_fragments = truncate_fragments(fragments, len(cif_data["_atom_site_label"]))
    #print(".. Done with filtrating the broken fragments.")
    
    # read into all fragments: molecules + residues
    xyz_fragments = extract_xyz_fragments(fragments, sc_rcoord)
    print(".. Done with obtaining xyz coordinates for the fragments.")
    
    print(".. Write xyz of all fragments into a text file.")
    with open("xyz_fragments_dict.py", "w") as f:
        f.write(str(xyz_fragments))
    
    write_xyz("supercell.xyz", xyz_dict2list(xyz_fragments))

    # Step 10:
    ## Calculate the center of mass for each monomers
    com_fragments = calc_center_of_mass(xyz_fragments)
    print(".. Done with calculating the centers of mass for each fragment.")
    
    # Step 11:
    ## Find the index of reference molecule in the supercell
    idx_refmol = find_refmol(com_fragments)
    print(".. Done with finding the reference molecule.")
    
    ## Calculate the distance of COMs between the reference molecule and other fragments
    dCOMs_other2ref = calc_dCOMs_other2ref(com_fragments, idx_refmol)
    print(".. Done with calculating distance of COMs between the reference molecule and others.")
    
    # Step 12:
    ## Generate the crystalline sphere around the reference molecule
    ## Make a new list of COM distance in there sphere
    ## Find the new index for the reference molecule
    (cryst_sphere, dCOMs_sphere, idx_refmol) = gen_cryst_sphere(r_thres, idx_refmol, dCOMs_other2ref, xyz_fragments)
    
    with open("cryst_sphere_dict.py", "w") as f:
        f.write("d = " + str(cryst_sphere))
        f.write("\n\n")
        f.write("idx_refmol = {}".format(int(idx_refmol)))
        
    with open("dCOMs_sphere_dict.py", "w") as f:
        f.write("d = " + str(dCOMs_sphere))
        f.write("\n\n")
        f.write("idx_refmol = {}".format(int(idx_refmol)))

    ## Check if all fragments in the crystalline sphere have the same number of atoms
    cryst_sphere_frags = list(cryst_sphere.values())
    isSimilar = all(len(element) == len(cryst_sphere_frags[0]) for element in cryst_sphere_frags)
    
    if (isSimilar): print("All molecules in the sphere are the same.")
    else:
        write_xyz("cryst_sphere.xyz", xyz_dict2list(cryst_sphere))
        print("There are still residues existing in the sphere.")
        print("\nEnd process with issue!\n")
        sys.exit()

    xyz_cryst_sphere = xyz_dict2list(cryst_sphere)
    write_xyz(atom_list = xyz_cryst_sphere, output_fname = "{}_sphere.xyz".format(casename))
    
    print("\nThere are {} molecules in the crystalline sphere.".format(len(cryst_sphere.keys())))          
    print("There are {} atoms in the crystalline sphere.\n".format(len(xyz_cryst_sphere)))
    print("\nThe index of reference molecule is {}.\n".format(idx_refmol))
    
    print(".. Done with generating the crystalline sphere.")
    
    # Step 13:
    ## Generate the dimer jobs
    idx_dimers = mol_in_dimers(r_dim_thres, dCOMs_sphere)
    ndimers = len(idx_dimers) - 1
    print("Number of dimers: {}".format(ndimers))
    gen_dimer_jobs(idx_dimers, idx_refmol, cryst_sphere)
    print(".. Done with generating the dimer jobs.")
    
    # Step 14:
    ## Generate the trimer jobs
    idx_trimers = mol_in_trimers(r_trim_thres, dCOMs_sphere)
    ntrimers = int((len(idx_trimers) - 1)*(len(idx_trimers) - 2)/2)
    print("Number of trimers: {}".format(ntrimers))
    gen_trimer_jobs(idx_trimers, idx_refmol, cryst_sphere)
    print(".. Done with generating the trimer jobs.")
    
    print("\nSucceeded in generating jobs for your lattice energy calculation!")
    
    end_time = time.time()
    print("The process took {} minute(s)\n".format(round((end_time - start_time)/60.0, 2)))
    
    sys.exit()
    

if __name__ == "__main__": 
    main(sys.argv[1])
