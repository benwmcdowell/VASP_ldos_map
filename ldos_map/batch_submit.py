import os
import getopt
import sys
import subprocess

#submits n jobs with values ranging from minval to maxval
def batch_submit_energies(fname,minval,maxval,n):
    erange=[[minval+i/(n-1),maxval+i/(n-1)] for i in range(n)]
    for i in range(n):
        with open(fname,'w+') as f:
            lines=f.readlines()
            for j in range(len(lines)):
                if 'srun' in lines[j]:
                    templine=lines[j].split()
                    for k in range(len(templine)):
                        if templine[k] in ['-e','--erange']:
                            templine[k+1]=','.join(erange[i])
                lines[j]=' '.join(templine)
        subprocess.call('sbatch {}'.format(fname), shell=True)




if __name__ == '__main__':
    fname='./map.sh'
    job_type=None
    n=1
    val_range=[0,1]
    try:
        opts,args=getopt.getopt(sys.argv[1:],'m:f:n:',['job_type=','filepath=','npoints='])
    except getopt.GetoptError:
        print('error in command line syntax')
        sys.exit(2)
    for i,j in opts:
        if i in ['-f','--filepath']:
            fname=j
        if i in ['-t','--job_type']:
            job_type=j
        if i in ['-n','--npoints']:
            n=int(j)
        if i in ['-v','--value_range']:
            val_range=[float(k) for k in j.split(',')]
            
    if not job_type:
        print('no batch submission type selected')
        sys.exit(2)
    elif job_type=='energy':
        batch_submit_energies(fname,val_range[0],val_range[1],n)
    else:
        print('invalid batch submission type selected')
        sys.exit(2)