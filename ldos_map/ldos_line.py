from numpy import array,dot,exp,linspace,where,zeros,shape,sqrt
from numpy.linalg import norm,inv
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.cm import bwr,ScalarMappable
from matplotlib.ticker import FormatStrFormatter
import getopt
from os.path import exists,getsize
from os import getcwd,chdir
from time import time
from pathos.multiprocessing import ProcessPool
from matplotlib.colors import Normalize,LinearSegmentedColormap
from lib import parse_doscar,parse_poscar,parse_CHGCAR,parse_bader_ACF,parse_potcar,tunneling_factor

class ldos_line:
    def __init__(self,filepath):
        self.npts=1
        self.emax=0
        self.emin=0
        self.estart=0
        self.eend=0
        self.x=array([[0.0 for i in range(self.npts)] for j in range(self.npts)])
        self.y=array([[0.0 for i in range(self.npts)] for j in range(self.npts)])
        self.z=array([[0.0 for i in range(self.npts)] for j in range(self.npts)])
        self.ldos=array([[0.0 for j in range(self.npts)] for i in range(self.npts)])
        self.exclude_args=['none']
        self.exclude=[]
        self.plot_atoms=[]
        self.nprocs=1
        self.periodic_coord=[]
        self.tip_disp=15.0
        self.unit_cell_num=4
        
        chdir(filepath)
        
    #reads in the ldos file created by self.write_ldos()
    def parse_ldos(self,filepath):
        self.tip_disp=float(filepath.split('_')[-1])
        with open(filepath,'r') as file:
            lines=file.readlines()
            self.npts=int(lines[0].split()[3])
            self.x=array([[0.0 for i in range(self.npts)] for j in range(self.npts)])
            self.y=array([[0.0 for i in range(self.npts)] for j in range(self.npts)])
            self.ldos=array([[0.0 for j in range(self.npts)] for i in range(self.npts)])
            self.emax=lines[1].split()[5]
            self.emin=lines[1].split()[3]
            for i in range(self.npts):
                for j in range(self.npts):
                    self.x[i][j]=lines[4+i].split()[j]
                    self.y[i][j]=lines[5+self.npts+i].split()[j]
                    self.ldos[i][j]=lines[6+2*self.npts+i].split()[j]
    
    #reads in the POSCAR and DOSCAR files
    def parse_VASP_output(self,**args):
        if 'doscar' in args:
            doscar=args['doscar']
        else:
            doscar='./DOSCAR'
            
        if 'poscar' in args:
            poscar=args['poscar']
        elif exists('./CONTCAR'):
            if getsize('./CONTCAR')>0:
                poscar='./CONTCAR'
            else:
                poscar='./POSCAR'
        else:
            poscar='./POSCAR'
                
        try:
            self.lv, self.coord, self.atomtypes, self.atomnums = parse_poscar(poscar)[:4]
            self.dos, self.energies, self.ef = parse_doscar(doscar)
        except:
            print('error reading input files')
            sys.exit()
        
    #sets up and performs the ldos interpolation based on the site projected DOS in DOSCAR
    def calculate_ldos(self,npts,emax,emin,lv_path,lv_origin,**args):
        self.emax=emax
        self.emin=emin
        self.npts=npts
        self.lv_path=dot(array(lv_path),self.lv)
        self.lv_origin=dot(array(lv_origin),self.lv)
        self.x=array([0.0 for i in range(self.npts)])
        self.y=array([0.0 for i in range(self.npts)])
        self.z=array([0.0 for i in range(self.npts)])
        if 'nprocs' in args:
            self.nprocs=int(args['nprocs'])
        if 'tip_disp' in args:
            self.tip_disp=float(args['tip_disp'])
            
        #the list exclude includes the indices of atoms to exlcude from LDOS integration
        self.exclude=[]
        if 'exclude' in args:
            self.exclude_args=args['exclude']
            counter=0
            for i in self.atomtypes:
                if i in args['exclude']:
                    for j in range(self.atomnums[self.atomtypes.index(i)]):
                        self.exclude.append(counter)
                        counter+=1
                else:
                    counter+=self.atomnums[self.atomtypes.index(i)]
        print(str(len(self.exclude))+' atoms excluded from LDOS averaging')
        
        if 'unit_cell_num' in args:
            self.unit_cell_num=args['unit_cell_num']
        
        for i in range(-1*self.unit_cell_num,self.unit_cell_num+1):
            for j in range(-1*self.unit_cell_num,self.unit_cell_num+1):
                for k in self.coord:
                    self.periodic_coord.append(k+self.lv[0]*i+self.lv[1]*j)
        self.periodic_coord=array(self.periodic_coord)
        
        for i in range(len(self.energies)):
            if self.energies[i]<emin:
                self.estart=i
            if self.energies[i]>emax:
                self.eend=i
                break
        else:
            print('specified emax exceeds maximum energy in DOSCAR.')
            print('integrating from {} to {} V'.format(self.emin,self.energies[-1]))
        
        self.ldos=array([[0.0 for i in range(self.eend-self.estart)] for j in range(self.npts)])
        
        if 'phi' in args and args['phi']!=0:
            self.K=array([tunneling_factor(i,args['phi']) for i in self.energies[self.estart:self.eend]])
        else:
            self.K=array([1.0 for i in range(self.estart-self.eend)])
            
        for i in range(self.npts):
            pos=array([0.0,0.0,max(self.coord[:,2])+self.tip_disp])
            pos+=self.lv_origin+self.lv_path*(i+0.5)/(self.npts)
            self.x[i], self.y[i] , self.z[i] = pos[0], pos[1], pos[2]
        start=time()
        #executes ldos integration in parallel on a ProcessPool of self.nprocs processors
        if self.nprocs>1:
            pool=ProcessPool(self.nprocs)
            output=pool.map(self.integrator, [i for i in range(self.npts) for j in range(self.npts)], [j for i in range(self.npts) for j in range(self.npts)])
            self.ldos=sum(output)
            pool.close()
        #executes ldos integration on a single processor
        else:
            for i in range(self.npts):
                pos=array([self.x[i],self.y[i],self.z[i]])
                counter=1
                for k in self.periodic_coord:
                    if counter==sum(self.atomnums)+1:
                            counter=1
                    if counter-1 not in self.exclude:
                        posdiff=norm(pos-k)
                        for l in range(len(self.dos[counter])):
                            self.ldos[i]+=self.dos[counter][l][self.estart:self.eend]*exp(-1.0*posdiff*self.K*1.0e-10)
                    counter+=1
        print('total time to integrate {} points: {} seconds on {} processors'.format(self.npts**2,time()-start,self.nprocs))
    
    #performs integration at single point of the x,y grid when run in parallel
    def integrator(self,i):
        from numpy import array
        pos=array([self.x[i],self.y[i],self.z[i]])
        temp_ldos=zeros((self.npts,self.eend-self.estart))
        counter=1
        for k in self.periodic_coord:
            if counter==sum(self.atomnums)+1:
                    counter=1
            if counter-1 not in self.exclude:
                posdiff=norm(pos-k)
                for l in range(len(self.dos[counter])):
                    temp_ldos[i]+=self.dos[counter][l][self.estart:self.eend]*exp(-1.0*posdiff*self.K*1.0e-10)
            counter+=1
        
        return temp_ldos
    
    #the ldos is written to a file in the current directory with the following format:
    #3 lines of informational header
    #1 blank line
    #self.npts lines each containing self.npts x values to define the grid
    #1 blank line
    #self.npts lines each containing self.npts y values to define the grid
    #1 blank line
    #self.npts lines each containing self.npts ldos values
    def write_ldos(self):
        filename='./map_from_{}-{}V_exclude_{}_disp_{}'.format(self.emin,self.emax,''.join(self.exclude_args),self.tip_disp)
        with open(filename, 'w') as file:
            file.write('DOS integrated over {} points per lattice vector'.format(self.npts))
            file.write('\nintegration performed from {} to {} V\n'.format(self.emin,self.emax))
            file.write('atoms types excluded from DOS integration: ')
            for i in self.exclude_args:
                file.write('{} '.format(i))
            file.write('\n\n')
            for axes in [self.x,self.y,self.ldos]:
                for i in range(self.npts):
                    for j in range(self.npts):
                        file.write(str(axes[i][j]))
                        file.write(' ')
                    file.write('\n')
                file.write('\n')
    
    #plots the ldos map and overlaid atoms on size+1 periodic cells
    def plot_map(self,**args):
        if 'cmap' in args:
            self.cmap=args['cmap']
            
        if 'normalize_ldos' in args:
            normalize_ldos=args['normalize_ldos']
        else:
            normalize_ldos=True
            
        self.ldosfig,self.ldosax=plt.subplots(1,1)
        
        path_distance=array([norm(self.lv_path*(i+0.5)/self.npts) for i in range(self.npts)])
        #plots the ldo
        if normalize_ldos:
            ldosmap=self.ldosax.pcolormesh(array([[path_distance[i] for j in range(self.eend-self.estart)] for i in range(self.npts)]),array([self.energies[self.estart:self.eend] for i in range(self.npts)]),self.ldos/max([max(i) for i in self.ldos]),cmap=self.cmap,shading='nearest')
        else:
            ldosmap=self.ldosax.pcolormesh(path_distance,self.energies[self.estart:self.eend],self.ldos,cmap=self.cmap,shading='nearest')
                
        if 'show_colorbar' in args:
            self.ldosfig.colorbar(ldosmap)
        
        self.ldosax.set(xlabel='distance along ldos line / $\AA$')
        self.ldosax.set(ylabel='energy - $E_f$ / eV')
        self.ldosax.set(title='LDOS line | {} $\AA$'.format(self.tip_disp))
        self.ldosfig.show()
        

if __name__=='__main__':
    sys.path.append(getcwd())
    exclude=['none']
    nprocs=1
    #a 15 Angstrom tip displacement gives realistic images at low voltage
    tip_disp=15.0
    #sets the number of unit cells considered along each lattice vector
    #4 gives good results when the sampling displacement is on the same order of magnitude as the lattice vector magnitudes
    unit_cell_num=4
    npts=1
    phi=0
    try:
        opts,args=getopt.getopt(sys.argv[1:],'e:n:x:p:t:u:w:',['erange=','npts=','exclude=','processors=','tip_disp=','num_unit_cells=','work_function='])
    except getopt.GetoptError:
        print('error in command line syntax')
        sys.exit(2)
    for i,j in opts:
        if i in ['-e','--erange']:
            emin=min([float(k) for k in j.split(',')])
            emax=max([float(k) for k in j.split(',')])
        if i in ['-n','--npts']:
            npts=int(j)
        if i in ['-x','--exclude']:
            exclude=[str(k) for k in j.split(',')]
        if i in ['-p', '--processors']:
            nprocs=int(j)
        if i in ['-t','--tip_disp']:
            tip_disp=float(j)
        if i in ['-u','--unit_cell_num']:
            unit_cell_num=int(j)
        if i in ['-w','-work_function']:
            phi=float(j)
    if exists('./DOSCAR'):
        main=ldos_map('./')
        main.parse_VASP_output()
        main.calculate_ldos(npts,emax,emin,exclude=exclude,nprocs=nprocs,tip_disp=tip_disp,unit_cell_num=unit_cell_num,phi=phi)
        main.write_ldos()