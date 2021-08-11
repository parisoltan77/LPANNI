#Programmed by Soltanzadeh
import json
import os
import networkx as nx
import pandas as pd 
from cdlib import NodeClustering, evaluation

def get_communities(path, first_delimiter, delimiter, seperate_comm_id):
    comm_dict = dict()
    with open(path, "rt") as f:
        for row in f:
            node, row = row.rstrip().split(first_delimiter, 1)
            node = int(node)
            record = list(map(int, row.split(delimiter)))
            for comm_id in record:
                if comm_id in comm_dict.keys():
                    comm_dict[comm_id].append(node)
                else:
                    comm_dict[comm_id] = [node]
    communities = []
    for k, v in comm_dict.items():
        if not seperate_comm_id:
            v.append(k)
        communities.append(v)
    return communities

def main():
    NMI_result = dict()
    Omega_result = dict()
    for N in [1000, 5000, 10000]:
        NMI_result[N] = dict()
        Omega_result[N] = dict()
        for mu in [0.1, 0.3]:
            NMI_result[N][mu] = {'NMI':[], 'NMI_max':[]}
            Omega_result[N][mu] = {'omega':[]}
            for om in range(2,9):
                communities = get_communities(os.path.join('N-{}-mu{:1.1f}-om{}-result.txt'.format(N, mu, om)), " ", " ", False)
                gt_communities = get_communities(os.path.join('N-{}-mu{:1.1f}-om{}-community.txt'.format(N, mu, om)), "\t", " ", True)
                df = pd.read_csv('N-{}-mu{:1.1f}-om{}.csv'.format(N, mu, om))
                G = nx.from_pandas_edgelist(df,source="From",target="To")
                coms = NodeClustering(communities, G, "", overlap=True)
                gt_coms = NodeClustering(gt_communities, G, "", overlap=True)
                nmi = evaluation.overlapping_normalized_mutual_information_LFK(coms, gt_coms)[0]
                nmi_max = evaluation.overlapping_normalized_mutual_information_MGH(coms, gt_coms)[0]
                omega = evaluation.omega(coms, gt_coms)[0]
                NMI_result[N][mu]['NMI'].append(nmi)
                NMI_result[N][mu]['NMI_max'].append(nmi_max)
                Omega_result[N][mu]['omega'].append(omega)

                #print("\nN: ", N,"mu: ", mu,"om: ", om,"NMI_result: ", NMI_result, "Omega_result: ", Omega_result)
        
        with open('LFR_nmi.json', 'w') as f:
            json.dump(NMI_result, f)
        with open('LFR_omega.json', 'w') as f:
            json.dump(Omega_result, f)
        

if __name__ == "__main__":
    main()

    