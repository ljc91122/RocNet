import torch
import os
from torch.utils import data
from scipy.io import loadmat
from enum import Enum

class OcTree(object):
    class NodeType(Enum):
        LEAF_FULL = 0  # full leaf node
        LEAF_EMPTY = 1 # empty leaf node
        LEAF_MIX = 2 # mixed leaf node
        NON_LEAF = 3 # non-leaf node

    class Node(object):
        def __init__(self, fea=None, child=None, node_type=None):
            self.fea = fea          # feature vector for a leaf node
            self.child = child        # child nodes
            self.node_type = node_type
            self.label = torch.LongTensor([self.node_type.value])

        def is_leaf(self):
            return self.node_type != OcTree.NodeType.NON_LEAF and self.fea is not None
        
        def is_empty_leaf(self):
            return self.node_type == OcTree.NodeType.LEAF_EMPTY

        def is_expand(self):
            return self.node_type == OcTree.NodeType.NON_LEAF

    def __init__(self, feas, ops):
        
        fea_list = [b.unsqueeze(0) for b in torch.split(feas, 1, 0)]
        fea_list.reverse()
        stack = []
        for id in range(ops.size()[1]):
            if ops[0, id] == OcTree.NodeType.LEAF_FULL.value:
                stack.append(OcTree.Node(fea=fea_list.pop(), node_type=OcTree.NodeType.LEAF_FULL))
            elif ops[0, id] == OcTree.NodeType.LEAF_EMPTY.value:
                stack.append(OcTree.Node(fea=fea_list.pop(), node_type=OcTree.NodeType.LEAF_EMPTY))
            elif ops[0, id] == OcTree.NodeType.LEAF_MIX.value:
                stack.append(OcTree.Node(fea=fea_list.pop(), node_type=OcTree.NodeType.LEAF_MIX))                
            elif ops[0, id] == OcTree.NodeType.NON_LEAF.value:
                child_node = []
                for i in range(8):
                    child_node.append(stack.pop())
                stack.append(OcTree.Node(child=child_node, node_type=OcTree.NodeType.NON_LEAF))

        assert len(stack) == 1
        self.root = stack[0]
        


class ROctDataset(data.Dataset):
    def __init__(self, dir, base, incre, transform=None):
        self.dir = dir
        
        fea_data = []
        op_data = []
        label_data = []
        
        self.trees = []

        for i in range(base,base+incre):
            
            if not os.path.exists(dir+'/fea_data'+str(i)+'.mat'):
                break
            
            try:

                fea_data = torch.from_numpy(loadmat(dir+'/fea_data'+str(i)+'.mat')['fea_all']).float()
                op_data = torch.from_numpy(loadmat(dir+'/op_data'+str(i)+'.mat')['ops']).int()

                fea = fea_data
                fea = fea.permute(3, 0, 1, 2)
                op = torch.t(op_data)
                tree = OcTree(fea,op)
                self.trees.append(tree)
            
            except:
                print("An exception occurred")


    def __getitem__(self, index):
        tree = self.trees[index]
        return tree

    def __len__(self):
        return len(self.trees)