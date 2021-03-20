from numpy import array,zeros,dot,sqrt,shape
from math import pi

#calculates factor for voltage (in eV) dependent tunneling probability
def tunneling_factor(V,E,phi):
    V*=1.60218e-19
    E*=1.60218e-19
    phi*=1.60218e-19
    prefactor=8/3/V*pi*sqrt(2*9.11e-31)/6.626e-34
    barrier=(phi-E+V)**(3/2)-(phi-E)**(3/2)
    return prefactor*barrier

#contains methods for reading VASP output files

#reads DOSCAR
def parse_doscar(filepath):
    with open(filepath,'r') as file:
        line=file.readline().split()
        atomnum=int(line[0])
        for i in range(5):
            line=file.readline().split()
        nedos=int(line[2])
        ef=float(line[3])
        dos=[]
        energies=[]
        for i in range(atomnum+1):
            if i!=0:
                line=file.readline()
            for j in range(nedos):
                line=file.readline().split()
                if i==0:
                    energies.append(float(line[0]))
                if j==0:
                    temp_dos=[[] for k in range(len(line)-1)]
                for k in range(len(line)-1):
                    temp_dos[k].append(float(line[k+1]))
            dos.append(temp_dos)
    energies=array(energies)-ef
    
    #orbitals contains the type of orbital found in each array of the site projected dos
    num_columns=shape(dos[1:])[1]
    if num_columns==3:
        orbitals=['s','p','d']
    elif num_columns==6:
        orbitals=['s_up','s_down','p_up','p_down','d_up','d_down']
    elif num_columns==9:
        orbitals=['s','p_y','p_z','p_x','d_xy','d_yz','d_z2-r2','d_xz','d_x2-y2']
    elif num_columns==18:
        orbitals=['s_up','s_down','p_y_up','p_y_down','p_z_up','p_z_down','p_x_up','p_x_down','d_xy_up','d_xy_down','d_yz_up','d_yz_down','d_z2-r2_up','d_z2-r2_down','d_xz_up','d_xz_down','d_x2-y2_up','d_x2-y2_down']
        
    #dos is formatted as [[total dos],[atomic_projected_dos for i in range(atomnum)]]
    #total dos has a shape of (4,nedos): [[spin up],[spin down],[integrated, spin up],[integrated spin down]]
    #atomic ldos have shapes of (6,nedos): [[i,j] for j in [spin up, spin down] for i in [s,p,d]]
    #energies has shape (1,nedos) and contains the energies that each dos should be plotted against
    return dos, energies, ef, orbitals

#reads POSCAR
def parse_poscar(ifile):
    with open(ifile, 'r') as file:
        lines=file.readlines()
        sf=float(lines[1])
        latticevectors=[float(lines[i].split()[j])*sf for i in range(2,5) for j in range(3)]
        latticevectors=array(latticevectors).reshape(3,3)
        atomtypes=lines[5].split()
        atomnums=[int(i) for i in lines[6].split()]
        if 'Direct' in lines[7] or 'Cartesian' in lines[7]:
            start=8
            mode=lines[7].split()[0]
        else:
            mode=lines[8].split()[0]
            start=9
            seldyn=[''.join(lines[i].split()[-3:]) for i in range(start,sum(atomnums)+start)]
        coord=array([[float(lines[i].split()[j]) for j in range(3)] for i in range(start,sum(atomnums)+start)])
        if mode!='Cartesian':
            for i in range(sum(atomnums)):
                for j in range(3):
                    while coord[i][j]>1.0 or coord[i][j]<0.0:
                        if coord[i][j]>1.0:
                            coord[i][j]-=1.0
                        elif coord[i][j]<0.0:
                            coord[i][j]+=1.0
                coord[i]=dot(coord[i],latticevectors)
            
    #latticevectors formatted as a 3x3 array
    #coord holds the atomic coordinates with shape ()
    try:
        return latticevectors, coord, atomtypes, atomnums, seldyn
    except NameError:
        return latticevectors, coord, atomtypes, atomnums

#reads the ACF file output by Bader analysis and returns contents
def parse_bader_ACF(ifile):
    with open(ifile, 'r') as file:
        x=[]
        y=[]
        z=[]
        charge=[]
        min_dist=[]
        vol=[]
        for i in range(2):
            line=file.readline()
        while True:
            line=file.readline().split()
            try:
                x.append(float(line[1]))
                y.append(float(line[2]))
                z.append(float(line[3]))
                charge.append(float(line[4]))
                min_dist.append(float(line[5]))
                vol.append(float(line[6]))
            #stops reading the file when '--------' is reached
            except IndexError:
                break
    
    return x, y, z, charge, min_dist, vol

#reads the number of valence electrons for each atom type for the POTCAR file
def parse_potcar(ifile):
    with open(ifile, 'r') as file:
        numvalence=[]
        counter=0
        while True:
            line=file.readline()
            if not line:
                break
            if counter==1:
                numvalence.append(float(line.split()[0]))
            if 'End of Dataset' in line:
                counter=-1
            counter+=1
        
    return numvalence

#reads the average electrostatic potential from LOCPOT
#in the case of spin polarized calculations, the average potential is returned
def parse_LOCPOT(ifile):
    with open(ifile,'r') as locpot:
        x=-1
        dim=0
        searching=True
        e=[]
        while searching:
            line=locpot.readline()
            if not line:
                break
            line=line.split()
            if len(line)==0:
                line=locpot.readline().split()
                x=0
                y=0
                z=0
                if dim==0:
                    dim=[int(i) for i in line]
                temp_e=zeros((dim[0],dim[1],dim[2]))
            elif x>-1:
                for i in line:
                    temp_e[x][y][z]=float(i)
                    x+=1
                    if x==dim[0]:
                        x=0
                        y+=1
                    if y==dim[1]:
                        y=0
                        z+=1
                    if z==dim[2]:
                        e.append(temp_e)
                        x=0
                        y=0
                        z=0
                        temp_e=zeros((dim[0],dim[1],dim[2]))
                        break
    e=sum(e)/len(e)
    return e
