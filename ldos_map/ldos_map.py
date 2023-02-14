from numpy import array,dot,exp,linspace,where,zeros,shape
from numpy.linalg import norm,inv
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.cm import bwr,ScalarMappable,jet
from matplotlib.ticker import FormatStrFormatter
import getopt
from os.path import exists,getsize
from os import getcwd,chdir
from time import time
from pathos.multiprocessing import ProcessPool
from matplotlib.colors import Normalize
from lib import parse_doscar,parse_poscar,parse_bader_ACF,parse_potcar,tunneling_factor
from math import ceil
from matplotlib.animation import FuncAnimation

class ldos_map:
    def __init__(self,filepath):
        self.npts=(1,1)
        self.emax=0
        self.emin=0
        self.estart=0
        self.eend=0
        self.x=array([[0.0 for i in range(self.npts[0])] for j in range(self.npts[1])])
        self.y=array([[0.0 for i in range(self.npts[0])] for j in range(self.npts[1])])
        self.z=array([[0.0 for i in range(self.npts[0])] for j in range(self.npts[1])])
        self.ldos=array([[0.0 for i in range(self.npts[0])] for j in range(self.npts[1])])
        self.exclude_args=['none']
        self.exclude=[]
        self.plot_atoms=[]
        self.nprocs=1
        self.periodic_coord=[]
        self.tip_disp=15.0
        self.unit_cell_num=4
        self.atom_colors=[]
        self.atom_sizes=[]
        self.cmap=plt.rcParams['image.cmap']
        self.charges=[]
        self.numvalence=[]
        self.ldosline=[]
        self.orbitals=[]
        self.sigma=0.0
        
        chdir(filepath)
    
    ### depreciated: current version uses parse_ldos() ###
    #reads in the ldos file created by self.write_ldos()
    def parse_ldos_old(self,filepath):
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
                    
    #reads in the ldos file created by self.write_ldos()
    def parse_ldos(self,filepath):
        header=filepath.split('_')
        if header[0][-3:]!='map':
            print('not a ldos map file. exiting...')
            sys.exit()
        
        erange=header[1][1:-1].split('to')
        self.emin=float(erange[0])
        self.emax=float(erange[1])
        self.tip_disp=float(header[2][1:])
        self.exclude=header[3][1:].split(',')
        if len(header[4][1:].split('x'))==1:
            self.npts=(int(header[4][1:]),int(header[4][1:]))
        else:
            self.npts=(int(header[4][1:].split('x')[0]),int(header[4][1:].split('x')[1]))
        self.phi=float(header[5][1:])
        self.unit_cell_num=int(header[6][1:])
        self.sigma=float(header[7][1:])
        
        with open(filepath,'r') as file:
            lines=file.readlines()
            self.orbitals=lines[2].split(', ')
            self.x=array([[0.0 for i in range(self.npts[0])] for j in range(self.npts[1])])
            self.y=array([[0.0 for i in range(self.npts[0])] for j in range(self.npts[1])])
            self.ldos=array([[[0.0 for j in range(self.npts[0])] for i in range(self.npts[1])] for k in range(len(self.orbitals))])
            for i in range(self.npts[1]):
                for j in range(self.npts[0]):
                    self.x[i][j]=lines[4+i].split()[j]
                    self.y[i][j]=lines[5+self.npts[1]+i].split()[j]
                    for k in range(len(self.orbitals)):
                        self.ldos[k][i][j]=lines[6+k+(2+k)*self.npts[1]+i].split()[j]
                    
    #the ldos is written to a file in the current directory with the following format:
    #3 lines of informational header
    #1 blank line
    #self.npts lines each containing self.npts x values to define the grid
    #1 blank line
    #self.npts lines each containing self.npts y values to define the grid
    #1 blank line
    #self.npts lines each containing self.npts ldos values
    def write_ldos(self):
        filename='./map_E{}to{}V_D{}_X{}_N{}_W{}_U{}_S{}'.format(self.emin,self.emax,self.tip_disp,','.join(self.exclude_args),'x'.join([str(i) for i in self.npts]),self.phi,self.unit_cell_num,self.sigma)
        with open(filename, 'w') as file:
            file.write('integration performed from {} to {} V\n'.format(self.emin,self.emax))
            file.write('atoms types excluded from DOS integration: ')
            for i in self.exclude_args:
                file.write('{} '.format(i))
            file.write('\norbital contributions to ldos: {}'.format(', '.join(self.orbitals)))
            file.write('\n\n')
            for axes in [self.x,self.y]:
                for i in range(self.npts[1]):
                    for j in range(self.npts[0]):
                        file.write(str(axes[i][j]))
                        file.write(' ')
                    file.write('\n')
                file.write('\n')
            for projection in self.ldos:
                for i in range(self.npts[1]):
                    for j in range(self.npts[0]):
                        file.write(str(projection[i][j]))
                        file.write(' ')
                    file.write('\n')
                file.write('\n')
    
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
        
        if 'potcar' in args:
            potcar=args['potcar']
        else:
            potcar='./POTCAR'
                
        try:
            self.lv, self.coord, self.atomtypes, self.atomnums = parse_poscar(poscar)[:4]
            self.dos, self.energies, self.ef, self.orbitals = parse_doscar(doscar)
            self.numvalence=parse_potcar(potcar)
        except:
            print('error reading input files')
            sys.exit()
        self.atom_colors=['black' for i in range(len(self.atomtypes))]
        self.atom_sizes=[200/max([norm(self.lv[j]) for j in range(2)]) for i in range(len(self.atomtypes))]
        
    #sets up and performs the ldos integration based on the site projected DOS in DOSCAR
    def calculate_ldos(self,npts,emax,emin,**args):
        self.emax=emax
        self.emin=emin
        self.npts=npts
        self.phi=args['phi']
        self.x=array([[0.0 for i in range(self.npts[0])] for j in range(self.npts[1])])
        self.y=array([[0.0 for i in range(self.npts[0])] for j in range(self.npts[1])])
        self.z=array([[0.0 for i in range(self.npts[0])] for j in range(self.npts[1])])
        self.ldos=array([[[0.0 for j in range(self.npts[0])] for i in range(self.npts[1])] for k in range(len(self.orbitals))])
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
            if self.energies[i]<self.emin:
                self.estart=i
            if self.energies[i]>self.emax:
                self.eend=i
                break
            
        if self.energies[0]>self.emin:
            self.estart=0
            self.emin=self.energies[0]
            print('specified emin is less than minimum energy in DOSCAR. setting emin to {}'.format(self.emax))
        if self.energies[-1]<self.emax:
            self.eend=len(self.energies)-1
            self.emax=self.energies[-1]
            print('specified emax exceeds maximum energy in DOSCAR. setting emax to {}'.format(self.emax))
                
        if self.phi!=0:
            self.K=array([tunneling_factor(self.emax,i,self.phi) for i in self.energies[self.estart:self.eend]])
        else:
            self.K=array([1.0 for i in range(self.estart-self.eend)])
            
        for i in range(self.npts[1]):
            for j in range(self.npts[0]):
                pos=array([0.0,0.0,max(self.coord[:,2])+self.tip_disp])
                pos+=self.lv[0]*(j+0.5)/(self.npts[0])+self.lv[1]*(i+0.5)/(self.npts[1])
                self.x[i][j], self.y[i][j] , self.z[i][j] = pos[0], pos[1], pos[2]
        start=time()
        #executes ldos integration in parallel on a ProcessPool of self.nprocs processors
        if self.nprocs>1:
            pool=ProcessPool(self.nprocs)
            output=pool.map(self.integrator, [i for i in range(self.npts[1]) for j in range(self.npts[0])], [j for i in range(self.npts[1]) for j in range(self.npts[0])])
            self.ldos=sum(output)
            pool.close()
        #executes ldos integration on a single processor
        else:
            for i in range(self.npts[1]):
                for j in range(self.npts[0]):
                    pos=array([self.x[i][j],self.y[i][j],self.z[i][j]])
                    counter=1
                    for k in self.periodic_coord:
                        if counter==sum(self.atomnums)+1:
                                counter=1
                        if counter-1 not in self.exclude:
                            posdiff=norm(pos-k)
                            sf=exp(-1.0*posdiff*self.K*1.0e-10)
                            for l in range(len(self.dos[counter])):
                                self.ldos[l][i][j]+=sum(self.dos[counter][l][self.estart:self.eend]*sf)
                        counter+=1
        print('total time to integrate {} points: {} seconds on {} processors'.format(self.npts[0]*self.npts[1],time()-start,self.nprocs))
    
    #performs integration at single point of the x,y grid when run in parallel
    def integrator(self,i,j):
        from numpy import array,zeros
        pos=array([self.x[i][j],self.y[i][j],self.z[i][j]])
        temp_ldos=zeros((len(self.orbitals),self.npts[1],self.npts[0]))
        counter=1
        for k in self.periodic_coord:
            if counter==sum(self.atomnums)+1:
                    counter=1
            if counter-1 not in self.exclude:
                posdiff=norm(pos-k)
                sf=exp(-1.0*posdiff*self.K*1.0e-10)
                for l in range(len(self.dos[counter])):
                    temp_ldos[l][i][j]+=sum(self.dos[counter][l][self.estart:self.eend]*sf)*(self.emax-self.emin)
            counter+=1
        
        return temp_ldos
    
    #applies a gaussian smear to the calculated ldos
    def smear_ldos(self,sigma):
        self.sigma=sigma
        
        if 'threshold' in args:
            threshold=int(args['threshold'])*sigma
        else:
            threshold=sigma*5

        tol=2*array([int(ceil(threshold*min(self.npts)/norm(self.lv[i]))) for i in range(2)])
        mask=zeros((1+tol[0]*2,1+tol[1]*2))
        for i in range(len(mask)):
            for j in range(len(mask[i])):
                pos=norm((i-tol[0])*self.lv[0]/self.npts[0]+(j-tol[1])*self.lv[1]/self.npts[1])
                if pos<threshold:
                    mask[i][j]=exp(-(pos)**2/2/sigma**2)
        
        #checks to see if self.ldos has already been summed for relevant orbital contributions
        #it is faster to call this function through the plot_map method, smearing the summed ldos
        if len(shape(self.ldos))==3:
            smeared_ldos=zeros((shape(self.ldos)[0],self.npts[1],self.npts[0]))
        else:
            smeared_ldos=zeros((self.npts[1],self.npts[0]))
        
        start=time()
        for i in range(self.npts[1]):
            for j in range(self.npts[0]):
                for k in range(i-tol[0],i+tol[0]+1):
                    m=k
                    for l in range(j-tol[1],j+tol[1]+1):
                        weight=mask[tol[0]-(i-m)][tol[1]-(j-l)]
                        if weight!=0:
                            while k>self.npts[1]-1 or k<0:
                                if k>self.npts[1]-1:
                                    k-=self.npts[1]
                                if k<0:
                                    k+=self.npts[1]
                            
                            while l>self.npts[0]-1 or l<0:
                                if l>self.npts[0]-1:
                                    l-=self.npts[0]
                                if l<0:
                                    l+=self.npts[0]
                            
                            if len(shape(self.ldos))==3:
                                for n in range(shape(self.ldos)[0]):
                                    smeared_ldos[n][i][j]+=weight*self.ldos[n][k][l]
                            else:
                                smeared_ldos[i][j]+=weight*self.ldos[k][l]
    
        print('total time for smearing: {} s'.format(time()-start))
        
        #renormalizes smeared ldos map relative to starting ldos
        if len(shape(self.ldos))==3:
            for n in range(shape(self.ldos)[0]):
                smeared_ldos[n]*=norm(self.ldos[n])/norm(smeared_ldos[n])
        else:
            smeared_ldos*=norm(self.ldos)/norm(smeared_ldos)
        self.ldos=smeared_ldos
        self.mask=mask
    
    #specifies which atoms to overlap on the ldos map
    #the argument ranges defines the range of atoms to include: [[xmin,xmax],[ymin,ymax],[zmin,zmax]]
    def overlay_atoms(self,ranges):
        for i in range(sum(self.atomnums)):
            for j in range(3):
                if self.coord[i][j] > max(ranges[j]) or self.coord[i][j] < min(ranges[j]):
                    break
            else:
                self.plot_atoms.append(i)
                
    #normalize ldos map to another ldos map object
    def normalize_to(self,ref):
        self.ldos/=ref.ldos
    
    #plots the ldos map and overlaid atoms on size[0]+1 by size[1]+1 periodic cells
    def plot_map(self,size,equalize_cbar=False,**args):
        if type(size)==int:
            size=(size,size)
        if 'cmap' in args:
            self.cmap=args['cmap']
        
        if 'orbitals' in args:
            orbitals_to_plot=args['orbitals']
        else:
            orbitals_to_plot=[i for i in range(len(self.orbitals))]
        if len(orbitals_to_plot)==len(self.orbitals):
            self.orbitals=['all']
        else:
            self.orbitals=[self.orbitals[i] for i in orbitals_to_plot]
        self.ldos=sum([self.ldos[i] for i in orbitals_to_plot])
        
        if 'smear' in args:
            self.smear_ldos(float(args['smear']))
            
        if 'normalize_ldos' in args:
            normalize_ldos=args['normalize_ldos']
        else:
            normalize_ldos=True
            
        if 'show_charges' in args:
            show_charges=args['show_charges']
            charge_list=[]
            for i in self.plot_atoms:
                for j in range(len(self.atomtypes)):
                    if i < sum(self.atomnums[:j+1]):
                        break
                if self.atomtypes[j] in show_charges or i in show_charges:
                    charge_list.append(self.charges[i]-self.numvalence[j])
            if not equalize_cbar:
                cnorm=Normalize(vmin=min(charge_list),vmax=max(charge_list))
            else:
                max_val=max([abs(min(charge_list)),abs(max(charge_list))])
                cnorm=Normalize(vmin=-max_val,vmax=max_val)
        else:
            show_charges=[]
        
        if 'clim' in args:
            clim=args['clim']
        else:
            clim=(0.0,1.0)
        
        self.ldosfig,self.ldosax=plt.subplots(1,1)
        
        #plots the ldos
        for i in range(size[0]+1):
            for j in range(size[1]+1):
                if normalize_ldos:
                    ldosmap=self.ldosax.pcolormesh(self.x+self.lv[0][0]*i+self.lv[1][0]*j,self.y+self.lv[0][1]*i+self.lv[1][1]*j,self.ldos/max([max(i) for i in self.ldos]),cmap=self.cmap,shading='nearest')
                else:
                    ldosmap=self.ldosax.pcolormesh(self.x+self.lv[0][0]*i+self.lv[1][0]*j,self.y+self.lv[0][1]*i+self.lv[1][1]*j,self.ldos,cmap=self.cmap,shading='nearest',vmin=clim[0],vmax=clim[1])
        
                    
        if 'show_colorbar' in args:
            map_cbar=self.ldosfig.colorbar(ldosmap)
            
        #holds the position and color of each atom
        tempx=[]
        tempy=[]
        colors=[]
        sizes=[]
        counter=0
        #plots the overlaid atoms as a scatterplot
        for i in self.plot_atoms:
            for j in range(len(self.atomtypes)):
                if i < sum(self.atomnums[:j+1]):
                    break
            for k in range(size[0]+1):
                for l in range(size[1]+1):
                    tempx.append(self.coord[i][0]+self.lv[0][0]*k+self.lv[1][0]*l)
                    tempy.append(self.coord[i][1]+self.lv[0][1]*k+self.lv[1][1]*l)
                    sizes.append(self.atom_sizes[j]/(max(size)+1))
                    if self.atomtypes[j] in show_charges or i in show_charges:
                        colors.append(bwr(cnorm(charge_list[counter])))
                    else:
                        colors.append(self.atom_colors[j])
            if self.atomtypes[j] in show_charges or i in show_charges:
                counter+=1
                        
        atom_scatter=self.ldosax.scatter(tempx,tempy,color=colors,s=sizes)
        self.ldosax.set(xlabel='x coordinate / $\AA$')
        self.ldosax.set(ylabel='y coordinate / $\AA$')
        self.ldosax.set(title='{} to {} V | {} $\AA$ | $\phi$ = {} | $\sigma$ = {}\ncontributing orbitals: {}'.format(self.emin, self.emax, self.tip_disp, self.phi, self.sigma, ', '.join(self.orbitals)))
        patches=[]
        for i in range(len(self.atomtypes)):
            if self.atomtypes[i] not in show_charges:
                patches.append(mpatches.Patch(color=self.atom_colors[i],label=self.atomtypes[i]))
                
        #if bader charges are plotted, a colorbar is displayed
        if len(show_charges)>0:
            cbar=self.ldosfig.colorbar(ScalarMappable(norm=cnorm,cmap=bwr))
            if type(show_charges[0])==str:
                cbar.set_label('net electron count of {}'.format(', '.join(show_charges)))
            cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%+.3f'))
            
        self.ldosfig.legend(handles=patches)
        self.ldosfig.subplots_adjust(top=0.9)
        self.ldosax.set_aspect('equal')
        self.ldosfig.show()
        
    def plot_ldos_slice(self,**args):
        self.slicefig,self.sliceax=plt.subplots(1,1)
        self.slicefig.show()
        
        if 'points' in args:
            points=args['points']
        else:
            points=200
        
        #if snap==True, the ldos line is made parallel to one of the lattice vectors
        if 'snap' in args:
            snap=True
            points=min(self.npts)
        else:
            snap=False
            
        results=self.ldosfig.ginput(2,timeout=60,show_clicks=True)
        for i in range(2):
            results[i]=array(results[i])
        if snap:
            projections=array([dot(results[1]-results[0],self.lv[i][:2]) for i in range(2)])
            max_index=where(projections==max(abs(projections)))[0][0]
            results[1]=results[0]+self.lv[max_index][:2]/norm(self.lv[max_index])*norm(results[1]-results[0])*projections[max_index]/abs(projections[max_index])
        positions=linspace(results[0],results[1],points)
        self.ldosax.plot(positions[:,0],positions[:,1])
        tempz=[0.0 for i in range(points)]
        for i in range(points):
            mindiff=max([norm(self.lv[j]) for j in range(3)])
            positions[i]=dot(positions[i],inv(self.lv[:2,:2]))
            for j in range(2):
                while positions[i][j]>1.0 or positions[i][j]<0.0:
                    if positions[i][j]>1.0:
                        positions[i][j]-=1.0
                    if positions[i][j]<0.0:
                        positions[i][j]+=1.0
            positions[i]=dot(positions[i],self.lv[:2,:2])
            for j in range(self.npts[1]):
                for k in range(self.npts[0]):
                    tempdiff=norm(positions[i]-array([self.x[j][k],self.y[j][k]]))
                    if mindiff>tempdiff:
                        mindiff=tempdiff
                        tempz[i]=self.ldos[j][k]
            self.ldosline.append(tempz)
            tempx=linspace(0,norm(results[1]-results[0]),points)
            self.sliceax.plot(tempx, tempz)
            self.sliceax.set(xlabel='position / $\AA$')
            plt.pause(.001)
            self.slicefig.canvas.draw()
            self.ldosfig.canvas.draw()
            
    def plot_bader_dos(self,**args):
        if 'types_to_plot' in args:
            types_to_plot=args['types_to_plot']
        else:
            types_to_plot=self.atomtypes
            
        if 'range_to_plot' in args:
            range_to_plot=args['range_to_plot']
        else:
            range_to_plot=[[] for i in range(3)]
            for i in range(3):
                tempvar=zeros(3)
                tempvar[i]+=1.0
                range_to_plot[i].append(-1.5*norm(dot(tempvar,self.lv)))
                range_to_plot[i].append(1.5*norm(dot(tempvar,self.lv)))
                
        self.baderdosfig,self.baderdosax=plt.subplots(1,1)
        
        atoms_to_plot=[]
        colorlist=[]
        for i in range(len(self.coord)):
            for j in range(len(self.atomtypes)):
                if i < sum(self.atomnums[:j+1]):
                    break    
            if self.atomtypes[j] in types_to_plot:
                for k in range(3):
                    if self.coord[i][k]<min(range_to_plot[k]) or self.coord[i][k]>max(range_to_plot[k]):
                        break
                else:
                    atoms_to_plot.append(i)
                    colorlist.append(self.numvalence[j]-self.charges[i])
                        
        cnorm=Normalize(vmin=min(colorlist),vmax=max(colorlist))
        for i,j in zip(atoms_to_plot,colorlist):
            tempy=zeros(len(self.energies))
            for k in range(len(self.dos[i+1])):
                tempy+=self.dos[i+1][k]
            self.baderdosax.plot(self.energies,tempy,color=bwr(cnorm(j)))
        self.baderdosax.set(xlabel='energy - $E_f$ / eV')
        self.baderdosax.set(ylabel='DOS / states $eV^{-1}$')
        self.baderdosax.set_facecolor('grey')
        cbar=plt.colorbar(ScalarMappable(norm=cnorm, cmap='bwr'))
        plt.set_cmap('bwr')
        cbar.set_label('net charge of {}'.format(', '.join(types_to_plot)))
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%+.3f'))
        plt.show()
        print('{} DOS curves plotted'.format(len(colorlist)))
                        
    #sets the color and size of atoms overlayed on the topography
    #by default, all projected atoms are black and equally sized
    def set_atom_appearance(self,colors,sizes):
        for i in range(len(self.atomtypes)):
            self.atom_colors[i]=colors[i]
            self.atom_sizes[i]=sizes[i]
    
    #reads in charges calculated by Bader analysis
    def overlay_bader_charges(self,**args):
        if 'filepath' in args:
            charges=parse_bader_ACF(args['filepath'])[3]
        else:
            charges=parse_bader_ACF('./ACF.dat')[3]
        self.charges=charges
        
#animates a series of ldos maps to show the evolution of spatial features with respect to some other variable (e.g. tip height, energy, etc)
def plot_moving_maps(plot_type,steps,directory,filepath,**args):
    if 'cmap' in args:
        cmap=args['cmap']
    else:
        cmap=plt.rcParams['image.cmap']
        
    if 'normalize_to' in args:
        ref=args['normalize_to']
    else:
        ref=False
        
    if 'animate_interval' in args:
        aint=args['animate_interval']
    else:
        aint=800
        
    if 'overlay_atoms' in args:
        overlay=args['overlay_atoms']
    else:
        overlay=[[-1000,1000],[-1000,1000],[4,1000]]
        
    if 'atom_appearance' in args:
        atom_appearance=args['atom_appearance']
    else:
        atom_appearance=['grey','pink','purple']
        
    if 'atom_size' in args:
        atom_size=args['atom_size']
    else:
        atom_size=[25,25,25]
        
    header=filepath.split('_')
    if header[0][-3:]!='map':
        print('not a ldos map file. exiting...')
        sys.exit()
    
    erange=header[1][1:-1].split('to')
    emin=float(erange[0])
    emax=float(erange[1])
    tip_disp=float(header[2][1:])
    exclude=header[3][1:].split(',')
    if len(header[4][1:].split('x'))==1:
        npts=(int(header[4][1:]),int(header[4][1:]))
    else:
        npts=(int(header[4][1:].split('x')[0]),int(header[4][1:].split('x')[1]))
    phi=float(header[5][1:])
    unit_cell_num=int(header[6][1:])
    sigma=float(header[7][1:])
    
    #plots the overlaid atoms as a scatterplot
    tempvar=ldos_map(directory)
    tempvar.parse_VASP_output()
    tempvar.overlay_atoms(overlay)
    tempvar.set_atom_appearance(atom_appearance,atom_size)
    atomx=[]
    atomy=[]
    atomsize=[]
    atomcolors=[]
    for i in tempvar.plot_atoms:
        for j in range(len(tempvar.atomtypes)):
            if i < sum(tempvar.atomnums[:j+1]):
                break
        atomx.append(tempvar.coord[i][0])
        atomy.append(tempvar.coord[i][1])
        atomsize.append(tempvar.atom_sizes[j])
        atomcolors.append(tempvar.atom_colors[j])
    
    xdata=zeros((npts[1],npts[0]))
    ydata=zeros((npts[1],npts[0]))
    zdata=zeros((len(steps),npts[1],npts[0]))
    for i in range(len(steps)):
        if plot_type=='energy':
            emin=steps[i][0]
            emax=steps[i][1]
        if plot_type=='height':
            tip_disp=steps[i]
        filename='./map_E{}to{}V_D{}_X{}_N{}_W{}_U{}_S{}'.format(emin,emax,tip_disp,','.join(exclude),'x'.join([str(i) for i in npts]),phi,unit_cell_num,sigma)
        tempvar=ldos_map(directory)
        tempvar.parse_ldos(filename)
        if ref:
            tempvar.normalize_to(ref)
        xdata=tempvar.x
        ydata=tempvar.y
        zdata[i]+=sum([tempvar.ldos[j,:,:] for j in range(shape(tempvar.ldos)[0])])
    
    fig,axs=plt.subplots(1,2,gridspec_kw={'width_ratios': [3,1]})
    fig.tight_layout()
    mesh=axs[0].pcolormesh(xdata,ydata,zdata[0],cmap=cmap,shading='nearest')
    pointer=axs[1].add_patch(mpatches.Rectangle((0,steps[0][0]),1,steps[0][1]-steps[0][0]))
    if len(shape(steps))==1:
        axs[1].set(xlim=(0,1),ylim=(steps[0],steps[-1]))
    else:
        axs[1].set(xlim=(0,1),ylim=(steps[0][0],steps[-1][1]))
    atomscatter=axs[0].scatter(atomx,atomy,s=atomsize,color=atomcolors)
    axs[0].set_aspect('equal')
    axs[1].yaxis.tick_right()
    axs[1].set_xticklabels([])
    axs[0].set(xlabel='position / $\AA$',ylabel='position / $\AA$')
    if plot_type=='energy':
        axs[1].set(ylabel='energy / eV')
    if plot_type=='height':
        axs[1].set(ylabel='height / $\AA$')
    axs[1].yaxis.set_label_position("right")
    
    def animate(i):
        for j in range(2):
            axs[j].cla()
        if len(shape(steps))==1:
            axs[1].set(xlim=(0,1),ylim=(steps[0],steps[-1]))
        else:
            axs[1].set(xlim=(0,1),ylim=(steps[0][0],steps[-1][1]))
        axs[0].set_aspect('equal')
        axs[1].yaxis.tick_right()
        axs[1].set_xticklabels([])
        axs[0].set(xlabel='position / $\AA$',ylabel='position / $\AA$')
        if plot_type=='energy':
            axs[1].set(ylabel='energy / eV')
        if plot_type=='height':
            axs[1].set(ylabel='height / $\AA$')
        mesh=axs[0].pcolormesh(xdata,ydata,zdata[i],cmap=cmap,shading='nearest')
        pointer=axs[1].add_patch(mpatches.Rectangle((0,steps[i][0]),1,steps[i][1]-steps[i][0],edgecolor='black',facecolor='grey'))
        atomscatter=axs[0].scatter(atomx,atomy,s=atomsize,color=atomcolors)
        return mesh,pointer,atomscatter

    anim=FuncAnimation(fig,animate,interval=aint,frames=len(steps),repeat=True,repeat_delay=1000)
    fig.show()
    return anim

if __name__=='__main__':
    sys.path.append(getcwd())
    exclude=['none']
    nprocs=1
    #a 15 Angstrom tip displacement gives realistic images at low voltage
    tip_disp=15.0
    #sets the number of unit cells considered along each lattice vector
    #4 gives good results when the sampling displacement is on the same order of magnitude as the lattice vector magnitudes
    unit_cell_num=4
    npts=(1,1)
    phi=0
    sigma=0.0
    try:
        opts,args=getopt.getopt(sys.argv[1:],'e:n:x:p:t:u:w:s:',['erange=','npts=','exclude=','processors=','tip_disp=','num_unit_cells=','work_function=','sigma='])
    except getopt.GetoptError:
        print('error in command line syntax')
        sys.exit(2)
    for i,j in opts:
        if i in ['-e','--erange']:
            emin=min([float(k) for k in j.split(',')])
            emax=max([float(k) for k in j.split(',')])
        if i in ['-n','--npts']:
            try:
                npts=(int(j.split(',')[0]),int(j.split(',')[1]))
            except IndexError:
                npts=(int(j.split(',')[0]),int(j.split(',')[0]))
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
        if i in ['-s','--sigma']:
            sigma=float(j)
    if exists('./DOSCAR'):
        main=ldos_map('./')
        main.parse_VASP_output()
        main.calculate_ldos(npts,emax,emin,exclude=exclude,nprocs=nprocs,tip_disp=tip_disp,unit_cell_num=unit_cell_num,phi=phi)
        if sigma!=0:
            main.smear_ldos(sigma)
        main.write_ldos()