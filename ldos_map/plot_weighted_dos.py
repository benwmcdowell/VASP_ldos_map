from numpy import array
import sys
import matplotlib.pyplot as plt
import getopt
from os.path import exists
from lib import parse_doscar,parse_poscar,tunneling_factor

def plot_weighted_dos(doscar,poscar,phi,**args):
    dos, energies, ef = parse_doscar(doscar)
    atomtypes, atomnums = parse_poscar(poscar)[2:4]
    
    if 'irange' in args and len(args['irange'])>0:
        integrate_dos=True
        irange=args['irange']
        erange=[]
        for i in range(len(energies)):
            if energies[i]>min(irange) and energies[i]<max(irange):
                erange.append(energies[i])
            elif energies[i]<min(irange):
                emin=i
            elif energies[i]>max(irange):
                emax=i
                break
    else:
        integrate_dos=False
        
    if 'nums' in args:
        nums=args['nums']
    else:
        nums=[]
    if 'types' in args:
        types=args['types']
    else:
        types=[]
    selected_atoms=[]
    if len(types)>0 and len(nums)>0:
        counter=1
        for i in atomtypes:
            if i in types:
                for j in range(atomnums[atomtypes.index(i)]):
                    if counter in nums:
                        selected_atoms.append(counter)
                    counter+=1
            else:
                counter+=atomnums[atomtypes.index(i)]
    elif len(nums)>0:
        selected_atoms=nums
    elif len(types)>0:
        counter=0
        for i in atomtypes:
            if i in types:
                for j in range(atomnums[atomtypes.index(i)]):
                    selected_atoms.append(counter)
                    counter+=1
            else:
                counter+=atomnums[atomtypes.index(i)]
    else:
        selected_atoms=[i for i in range(sum(atomnums))]
        
    K=array([tunneling_factor(i,phi) for i in energies])
    
    plt.figure()
    for i in selected_atoms:
        for j in range(len(atomnums)):
            if i<sum(atomnums[:j+1]):
                atomlabel=atomtypes[j]
                break
        if not integrate_dos:
            tempy=array([0.0 for j in range(len(energies))])
            for j in range(len(dos[i+1])):
                tempy+=dos[i+1][j]*K
            plt.plot(energies,tempy,label='{} #{}'.format(atomlabel,i-sum(atomnums[:atomtypes.index(atomlabel)])))
        else:
            tempy=array([0.0 for j in range(len(erange))])
            for j in range(len(dos[i+1])):
                tempy+=dos[i+1][j][emin+1:emax]*K[emin+1:emax]
            for j in range(1,len(tempy)):
                tempy[j]+=tempy[j-1]
            plt.plot(erange,tempy,label='{} #{}'.format(atomlabel,i-sum(atomnums[:atomtypes.index(atomlabel)])))
    plt.xlabel('energy - $E_f$ / eV')
    if not integrate_dos:
        plt.ylabel('DOS / states $eV^{-1}$')
    else:
        plt.ylabel('integrated DOS / # states')
    plt.legend()
    plt.show()
 
if __name__=='__main__':
    irange=[]
    doscar='./DOSCAR'
    if exists('./CONTCAR'):
        poscar='./CONTCAR'
    else:
        poscar='./POSCAR'
    atomnums=[]
    atomtypes=[]
    try:
        opts,args=getopt.getopt(sys.argv[1:],'a:t:i:hp:',['atomnums=','types=','integrated=','help','phi='])
    except getopt.GetoptError:
        print('error in command line syntax')
        sys.exit(2)
    for i,j in opts:
        if i in ['-a','--atomnums']:
            atomnums=[int(k) for k in j.split(',')]
        if i in ['-t','--types']:
            atomtypes=[str(k) for k in j.split(',')]
        if i in ['-i','--integrated']:
            irange=[float(k) for k in j.split(',')]
        if i in ['-h','--help']:
            print('''
required arguments:
-p, --phi               specify the work function, which is used to weight the DOS by the probability of tunneling into a state at its corresponding energy
    
plotting options:
-a, --atomnums          specify site projected DOS to plot by the index of atoms: 1,2,3,etc...
-t, --types             specify which site projected DOS to plot by atom type: Au,C,etc...
-i, --integrated        integrate the DOS between the range specified. ie -i 0,3 will plot the integrated DOS from 0 to 3 eV above the Fermi level

help options:
-h, --help               display this help message
                  ''')
            sys.exit()
    if exists(doscar):
        plot_weighted_dos(doscar,poscar,nums=atomnums,types=atomtypes,irange=irange)
