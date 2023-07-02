import numpy as np
from trace import *
from xcluster.models.PNode import PNode

class WTSampling:
    def __init__(self, dataFile):
        self.dataFile = dataFile
        dataSet = []
        
        with open(dataFile, 'r') as fp:
            data = fp.readline()
            while data != '':
                dataSet.append(data)
            self.dataSet = dataSet
            
        
    def build_sliding_tree(max_leaves, exact_dist_thres=10):
        root = PNode(exact_dist_thres=exact_dist_thres)

        count = 0
        for pt in self.dataSet:
            if count < max_leaves:
                root = root.insert(pt)
                count += 1
            else:
                self.delete_unlikely_node()
                root = root.insert(pt)

        self.tree = root

        
    def delete_unlikely_node(self):
        node = self.root

        while not node.is_leaf():
            node = max(node.children, key=lambda x: x.point_counter)

        sibling = node.siblings()[0]
        parent = node.parent
        parent.children = []
        parent.pts = sibling.pts
        parent.point_counter = sibling.point_counter    
        sibling.deleted = True
        node.deleted = True

        parent._update_params_recursively()
        
    def sampling(num, seed = None):
        np.random.seed(seed)
        samples = set()
        count = 0
        while count < num:
            node = self.root
            while not node.is_leaf():
                node = np.random.choice(node.children, 1)[0]
            if node not in samples:
                samples.add(node)
                count += 1
        return samples
    

def traceEncoding(traces, spanMaps, index, savePath):
    with open(savePath, 'w') as fp:
        for trace in traces:
            spanCount = [0] * len(spanMaps)
            for span in trace.spans:
                label = getSpanLabel(span)
                if label not in spanMaps:
                    spanMaps[label] = len(spanMaps)
                    spanCount.append(1)
                    continue
                spanCount[spanMaps[label]] += 1
            abnormal = 1 if trace.abnormal else 0
            encoding = '%d %d %s\n' % (index, abnormal, ' '.join([str(n) for n in spanCount]))
            fp.write(encoding)
            index += 1
    return index