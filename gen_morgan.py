"""A total of 21 Morgan fingerprints will be generated:
        Unhashed,
        128 bit,
        256 bit,
        512 bit,
        1024 bit,
        2048 bit
        4096 bit;

        with radii of:
        1,
        2,
        3.

A total of 21xfiles svmlight files (in csv format).
"""

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
from rdkit.six import StringIO
import re

# Define functions to sort by identifier value, which is required for svmlight-format.
def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

files = ['Datasets/nr-ahr_nosalt.sdf.std_nodupl_class.sdf',
         'Datasets/nr-er_nosalt.sdf.std_nodupl_class.sdf',
         'Datasets/smiles_cas_N6512.sdf.std_class.sdf',
         'Datasets/sr-mmp_nosalt.sdf.std_nodupl_class.sdf']

count = 0
for f in files:
    mols = Chem.SDMolSupplier(f)

    # these lines are needed to obtain class values (-1 or 1) from the SDFs
    sio = StringIO()
    w = Chem.SDWriter(sio)
    for m in mols: w.write(m)
    w.flush()

    # HASHED FINGERPRINTS
    for i in 128,256,512,1024,2048,4096:
        for j in 1,2,3:
            fps = [AllChem.GetMorganFingerprintAsBitVect(m, j, i) for m in mols]

            y_label = []
            # exchange "<class>" for "<active>" if SDFs use "<active>" instead of "<class>" for activity data
            if len(re.findall('<class>.*\n-?[1]\n', sio.getvalue())) > 0:
                for m in re.findall('<class>.*\n-?[1]\n', sio.getvalue()):
                    m = m.split('\n')
                    label = m[-2]
                    y_label.append(label)

            a = []
            b = []
            for fp in fps:
                array = np.zeros((1,))
                DataStructs.ConvertToNumpyArray(fp, array)
                index = list(range(i))
                lst  = list(zip(array, index))
                for element_tuple in lst:
                    if element_tuple[0] == 1:
                        a.append(str(element_tuple[1])+':1')
                b.append(a)
                a = []

            df = pd.DataFrame(b)
            df.insert(loc=0, column='Class', value=y_label)
            df.to_csv(str(f)+'_svmlight-format_hashed_%(1)s_r_%(2)s'% {'1':i, '2':j} +'.csv', sep="\t", header=False, index=False)
            count += 1
            print(str(count)+'/84'+ " done")
    # HASHED DONE

    # UNHASHED FINGERPRINTS
    for r in 1,2,3:
        fps = [AllChem.GetMorganFingerprint(m, r) for m in mols]

        y_label = []
        if len(re.findall('<class>.*\n-?[1]\n', sio.getvalue())) > 0:
            for m in re.findall('<class>.*\n-?[1]\n', sio.getvalue()):
                m = m.split('\n')
                label = m[-2]
                y_label.append(label)


        # Create an index value for each unqiue identifier
        a = []
        b = []
        for fp in fps:
            x = fp.GetNonzeroElements()
            for i in x.keys():
                a.append(i)
        a = list(set(a))
        b = list(range(len(a)))
        c = dict(zip(a,b))

        # Create unhashed with new smaller values as identifiers
        d = []
        g = []
        for fp in fps:
            x = fp.GetNonzeroElements()
            for key, value in x.items():
                y = str(c[key])+':'+str(value)
                d.append(y)
                d.sort(key=natural_keys)
            g.append(d)
            d = []

        df = pd.DataFrame(g)
        df.insert(loc=0, column='Class', value=y_label)
        df.to_csv(str(f)+'_svmlight-format_unhashed_r_%s'%r +'.csv', sep="\t", header=False, index=False)
        count += 1
        print(str(count)+'/84'+ " done")
    # UNHASHED DONE
