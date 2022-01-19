import os
import getopt
import sys
import subprocess

#submits n jobs with values ranging from minval to maxval
def batch_submit_energies(fname,emin,emax,n,de):
    erange=[[emin+de*i,emax+de*i] for i in range(n)]
    for i in range(n):
        with open(fname,'r') as f:
            lines=f.readlines()
            for j in range(len(lines)):
                if 'srun' in lines[j]:
                    templine=lines[j].split()
                    for k in range(len(templine)):
                        if templine[k] in ['-e','--erange']:
                            templine[k+1]=','.join([str(round(erange[i][l],1)) for l in range(2)])
                    lines[j]=' '.join(templine)
                
        with open(fname,'w+') as f:
            for j in lines:
                f.write(j)
                
        subprocess.call('sbatch {}'.format(fname), shell=True)




if __name__ == '__main__':
    fname='./map.sh'
    job_type=None
    n=1
    val_range=[0,1]
    de=0
    try:
        opts,args=getopt.getopt(sys.argv[1:],'j:f:n:v:d:',['job_type=','filepath=','npoints=','value_range==','step='])
    except getopt.GetoptError:
        print('error in command line syntax')
        sys.exit(2)
    for i,j in opts:
        if i in ['-f','--filepath']:
            fname=j
        if i in ['-j','--job_type']:
            job_type=j
        if i in ['-n','--npoints']:
            n=int(j)
        if i in ['-v','--value_range']:
            val_range=[float(k) for k in j.split(',')]
        if i in ['-d','--step']:
            de=float(j)
            
    if not job_type:
        print('no batch submission type selected')
        sys.exit(2)
    elif job_type=='energy':
        batch_submit_energies(fname,val_range[0],val_range[1],n,de)
    else:
        print('invalid batch submission type selected')
        sys.exit(2)