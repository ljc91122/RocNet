import math
import torch
from torch import nn
from torch.autograd import Variable
from time import time

#########################################################################################
## Encoder
#########################################################################################
class Sampler(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(Sampler, self).__init__()
        self.conv1 = nn.Conv3d(64, feature_size, kernel_size=4, stride=1, bias=True)
        self.conv2 = nn.Conv3d(128, feature_size, kernel_size=4, stride=2, bias=True)
        
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2mu = nn.Linear(hidden_size, feature_size)
        self.mlp2var = nn.Linear(hidden_size, feature_size)
        self.tanh = nn.Tanh()
        self.tanh = nn.ELU()
#         self.bn1 = nn.BatchNorm3d(128)
        self.bn2 = nn.BatchNorm3d(feature_size)
#         self.bn3 = nn.BatchNorm1d(hidden_size)
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data = nn.init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu')) 
        
    def forward(self, input):
        #print(input.size())
        output = self.tanh(self.conv1(input))
        #output = self.tanh(self.bn2(self.conv2(output)))
        
        output = output.view(-1, output.size()[1])#
#         encode = self.mlp1(output)
#         mu = self.mlp2mu(encode)
        
#         logvar = self.mlp2var(encode)
#         std = logvar.mul(0.5).exp_() # calculate the STDEV
#         eps = Variable(torch.FloatTensor(std.size()).normal_().cuda()) # random normalized noise
#         KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
#         return torch.cat([eps.mul(std).add_(mu), KLD_element], 1)
        return output



class LeafEncoder(nn.Module):

    def __init__(self, input_size, feature_size):
        super(LeafEncoder, self).__init__()
        self.encoder = nn.Linear(feature_size, feature_size)
        
        
        #3d conv layer1
        self.conv1 = nn.Conv3d(1, 16, kernel_size=4, stride=2, bias=True, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        
        
        #3d conv layer2
        self.conv2 = nn.Conv3d(16, 32, kernel_size=4, stride=2, bias=True, padding=1)
        self.bn2 = nn.BatchNorm3d(32)

        #3d conv layer3
        self.conv3 = nn.Conv3d(32, 64, kernel_size=4, stride=2, bias=True, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        
#         #3d conv layer4
#         self.conv4 = nn.Conv3d(64, feature_size, kernel_size=4, stride=1, bias=True)
#         self.bn4 = nn.BatchNorm3d(feature_size)
        
        
        #self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.tanh = nn.Tanh()
        self.tanh = nn.ELU()
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data = nn.init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu')) 

    def forward(self, leaf_input):

        #leaf_vector = self.conv1(leaf_input.add(-0.5).mul(2))
        leaf_vector = self.conv1(leaf_input)
        leaf_vector = self.bn1(leaf_vector)
        leaf_vector = self.tanh(leaf_vector)

        #print('input leaf')
        #print(leaf_input.size())
        
        leaf_vector = self.conv2(leaf_vector)
        leaf_vector = self.bn2(leaf_vector)
        leaf_vector = self.tanh(leaf_vector)
        #leaf_vector = self.tanh(leaf_vector)
        
        #print('conv1 leaf')
        #print(leaf_vector.size())
        
        leaf_vector = self.conv3(leaf_vector)
        leaf_vector = self.bn3(leaf_vector)
        leaf_vector = self.tanh(leaf_vector)
        #leaf_vector = self.tanh(leaf_vector)
        
        #print('conv2 leaf')
        #print(leaf_vector.size())
        
#         leaf_vector = self.conv4(leaf_vector)
#         leaf_vector = self.bn4(leaf_vector)
#         leaf_vector = self.tanh(leaf_vector)
#         #leaf_vector = self.tanh(leaf_vector)
        
        #leaf_vector = leaf_vector.squeeze()

        return leaf_vector
    
class LeafEncoder2(nn.Module):

    def __init__(self, input_size, feature_size):
        super(LeafEncoder2, self).__init__()
        
        self.feature_size = feature_size


    def forward(self, leaf_input):

#         #print(leaf_input)
#         #print(torch.zeros(leaf_input.size()[0],self.feature_size))
        return Variable(torch.zeros(leaf_input.size()[0],64,4,4,4).cuda())

class NodeEncoder(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(NodeEncoder, self).__init__()
        self.child1 = nn.Conv3d(64, 128, kernel_size=1, stride=1, bias=True)
        self.child2 = nn.Conv3d(64, 128, kernel_size=1, stride=1, bias=False)
        self.child3 = nn.Conv3d(64, 128, kernel_size=1, stride=1, bias=False)
        self.child4 = nn.Conv3d(64, 128, kernel_size=1, stride=1, bias=False)
        self.child5 = nn.Conv3d(64, 128, kernel_size=1, stride=1, bias=False)
        self.child6 = nn.Conv3d(64, 128, kernel_size=1, stride=1, bias=False)
        self.child7 = nn.Conv3d(64, 128, kernel_size=1, stride=1, bias=False)
        self.child8 = nn.Conv3d(64, 128, kernel_size=1, stride=1, bias=False)

        self.second = nn.Conv3d(128, 64, kernel_size=3, stride=1, bias=True, padding=1)

        self.tanh = nn.Tanh()
        self.tanh = nn.ELU()
        
        self.bn1 = nn.BatchNorm3d(128)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn3 = nn.BatchNorm3d(128)
        self.bn4 = nn.BatchNorm3d(128)
        self.bn5 = nn.BatchNorm3d(128)
        self.bn6 = nn.BatchNorm3d(128)
        self.bn7 = nn.BatchNorm3d(128)
        self.bn8 = nn.BatchNorm3d(128)
        
        self.bn11 = nn.BatchNorm3d(128)
        self.bn12 = nn.BatchNorm3d(64)
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data = nn.init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu')) 

    def forward(self, c1,c2,c3,c4,c5,c6,c7,c8):
        output = self.bn1(self.child1(c1))
        output += self.bn2(self.child2(c2))
        output += self.bn3(self.child3(c3))
        output += self.bn4(self.child4(c4))
        output += self.bn5(self.child5(c5))
        output += self.bn6(self.child6(c6))
        output += self.bn7(self.child7(c7))
        output += self.bn8(self.child8(c8))

        
        output = self.bn11(output)
        
        output = self.tanh(output)
        output = self.second(output)
        #print(output.size())
        if len(output.size())==1:
            output = output.unsqueeze(0)
        output = self.bn12(output)
        output = self.tanh(output)
#         output = self.third(output)
#         output = self.tanh(output)
        
        return output


class ROctEncoder(nn.Module):

    def __init__(self, config):
        super(ROctEncoder, self).__init__()
        self.leaf_encoder = LeafEncoder(input_size = config.leaf_code_size, feature_size = config.feature_size)
        self.leaf_encoder2 = LeafEncoder2(input_size = config.leaf_code_size, feature_size = config.feature_size)
        
        self.node_encoder1 = NodeEncoder(feature_size = config.feature_size, hidden_size = config.hidden_size)
        self.node_encoder2 = NodeEncoder(feature_size = config.feature_size, hidden_size = config.hidden_size)
        self.node_encoder3 = NodeEncoder(feature_size = config.feature_size, hidden_size = config.hidden_size)
        self.node_encoder4 = NodeEncoder(feature_size = config.feature_size, hidden_size = config.hidden_size)
        self.node_encoder5 = NodeEncoder(feature_size = config.feature_size, hidden_size = config.hidden_size)
        
        self.sample_encoder = Sampler(feature_size = config.feature_size, hidden_size = config.hidden_size)


    def LeafEncoder(self, fea):
        return self.leaf_encoder(fea)

    def LeafEncoder2(self, fea):
        return self.leaf_encoder2(fea)
    
    def NodeEncoder1(self, c1,c2,c3,c4,c5,c6,c7,c8):
        return self.node_encoder1(c1,c2,c3,c4,c5,c6,c7,c8)
    
    def NodeEncoder2(self, c1,c2,c3,c4,c5,c6,c7,c8):
        return self.node_encoder2(c1,c2,c3,c4,c5,c6,c7,c8)
    
    def NodeEncoder3(self, c1,c2,c3,c4,c5,c6,c7,c8):
        return self.node_encoder3(c1,c2,c3,c4,c5,c6,c7,c8)
    
    def NodeEncoder4(self, c1,c2,c3,c4,c5,c6,c7,c8):
        return self.node_encoder4(c1,c2,c3,c4,c5,c6,c7,c8)    
    
    def NodeEncoder5(self, c1,c2,c3,c4,c5,c6,c7,c8):
        return self.node_encoder5(c1,c2,c3,c4,c5,c6,c7,c8)   
    
    def sampleEncoder(self, feature):
        return self.sample_encoder(feature)

def encode_structure_fold(fold, tree):

    def encode_node(node,l):
        if node.is_leaf():
            if not node.is_empty_leaf():
                return fold.add('LeafEncoder', node.fea)
            else:
                ##print(node.fea)
                return fold.add('LeafEncoder2', node.fea)
        elif node.is_expand():
            child = []
            for i in range(8):
                child.append(encode_node(node.child[i],l+1))
            return fold.add('NodeEncoder'+str(l), child[0], child[1],child[2],child[3],child[4],child[5],child[6],child[7])

    encoding = encode_node(tree.root,1)
    return fold.add('sampleEncoder', encoding)
    #return encoding

#########################################################################################
## Decoder
#########################################################################################

class SampleDecoder(nn.Module):
    """ Decode a randomly sampled noise into a feature vector """
    def __init__(self, feature_size, hidden_size):
        super(SampleDecoder, self).__init__()
        
        self.deconv1 = nn.ConvTranspose3d(feature_size, 64, kernel_size=4, stride=1, bias=True)
        self.deconv2 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, bias=True, padding=1)
        
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        #self.bn1 = nn.BatchNorm1d(hidden_size)
        self.mlp2 = nn.Linear(hidden_size, feature_size)
#         self.bn2 = nn.BatchNorm1d(feature_size)
#         self.bn3 = nn.BatchNorm3d(128)
        self.bn4 = nn.BatchNorm3d(64)
        self.tanh = nn.Tanh()
        self.tanh = nn.ELU()
        
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d):
                m.weight.data = nn.init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu')) 
        
    def forward(self, input_feature):
#         output = self.mlp1(input_feature)
#         output = self.mlp2(output)
        output = input_feature.view(-1, input_feature.size()[1] , 1, 1, 1)#
        
        output = self.tanh(self.deconv1(output))
#         #output = self.tanh(self.bn4(self.deconv2(output)))
        
        return output


class NodeClassifier(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(NodeClassifier, self).__init__()
        
        self.conv1 = nn.Conv3d(64, feature_size, kernel_size=4, stride=1, bias=True)
        self.conv2 = nn.Conv3d(128, feature_size, kernel_size=4, stride=2, bias=True)
        
        
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.tanh = nn.Tanh()
        self.tanh = nn.ELU()
        self.mlp2 = nn.Linear(hidden_size, 4)
        self.bn = nn.BatchNorm1d(hidden_size)
        
        self.bn1 = nn.BatchNorm3d(128)
        self.bn2 = nn.BatchNorm3d(feature_size)
        
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d):
                m.weight.data = nn.init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu')) 

    def forward(self, input_feature):
        
        output = self.conv1(input_feature)
        #print(output.size())
        #output = self.tanh(self.bn2(self.conv2(output)))
        
        output = self.mlp1(output.view(-1, output.size()[1]))
        output = self.tanh(output)
        output = self.mlp2(output)
        #output = self.softmax(output)
        return output

class NodeDecoder(nn.Module):
    """ Decode an input (parent) feature into a left-child and a right-child feature """
    def __init__(self, feature_size, hidden_size):
        super(NodeDecoder, self).__init__()
        self.mlp = nn.ConvTranspose3d(64, 128, kernel_size=3, stride=1, bias=True, padding=1)
        self.mlp_child1 = nn.ConvTranspose3d(128, 64, kernel_size=1, stride=1, bias=True)
        self.mlp_child2 = nn.ConvTranspose3d(128, 64, kernel_size=1, stride=1, bias=True)
        self.mlp_child3 = nn.ConvTranspose3d(128, 64, kernel_size=1, stride=1, bias=True)
        self.mlp_child4 = nn.ConvTranspose3d(128, 64, kernel_size=1, stride=1, bias=True)
        self.mlp_child5 = nn.ConvTranspose3d(128, 64, kernel_size=1, stride=1, bias=True)
        self.mlp_child6 = nn.ConvTranspose3d(128, 64, kernel_size=1, stride=1, bias=True)
        self.mlp_child7 = nn.ConvTranspose3d(128, 64, kernel_size=1, stride=1, bias=True)
        self.mlp_child8 = nn.ConvTranspose3d(128, 64, kernel_size=1, stride=1, bias=True)
        
        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(64)
        self.bn4 = nn.BatchNorm3d(64)
        self.bn5 = nn.BatchNorm3d(64)
        self.bn6 = nn.BatchNorm3d(64)
        self.bn7 = nn.BatchNorm3d(64)
        self.bn8 = nn.BatchNorm3d(64)
        
        self.tanh = nn.Tanh()
        self.tanh = nn.ELU()
        self.bn = nn.BatchNorm3d(128)
        
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d):
                m.weight.data = nn.init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu')) 
        

    def forward(self, parent_feature):
        
        #print(parent_feature.size())
        vector = self.mlp(parent_feature)
        
        vector = self.bn(vector)
        vector = self.tanh(vector)
        
        child_feature1 = self.bn1(self.mlp_child1(vector))
        child_feature1 = self.tanh(child_feature1)
        child_feature2 = self.bn2(self.mlp_child2(vector))
        child_feature2 = self.tanh(child_feature2)
        child_feature3 = self.bn3(self.mlp_child3(vector))
        child_feature3 = self.tanh(child_feature3)
        child_feature4 = self.bn4(self.mlp_child4(vector))
        child_feature4 = self.tanh(child_feature4)
        child_feature5 = self.bn5(self.mlp_child5(vector))
        child_feature5 = self.tanh(child_feature5)
        child_feature6 = self.bn6(self.mlp_child6(vector))
        child_feature6 = self.tanh(child_feature6)
        child_feature7 = self.bn7(self.mlp_child7(vector))
        child_feature7 = self.tanh(child_feature7)
        child_feature8 = self.bn8(self.mlp_child8(vector))
        child_feature8 = self.tanh(child_feature8)
        
        return  child_feature1,child_feature2,child_feature3,child_feature4,child_feature5,child_feature6,child_feature7,child_feature8

class leafDecoder(nn.Module):
    
    def __init__(self, feature_size, leaf_size):
        super(leafDecoder, self).__init__()
        self.encoder = nn.Linear(feature_size, feature_size)
        
        
#         #3d deconv layer1
#         self.conv1 = nn.ConvTranspose3d(feature_size, 64, kernel_size=4, stride=2, bias=True, padding=1)
#         self.bn1 = nn.BatchNorm3d(64)
        
        
        #3d deconv layer2
        self.deconv2 = nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, bias=True, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        
        #3d deconv layer3
        self.deconv3 = nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, bias=True, padding=1)
        self.bn3 = nn.BatchNorm3d(16)
        
        #3d deconv layer4
        self.deconv4 = nn.ConvTranspose3d(16, 1, kernel_size=4, stride=2, bias=True, padding=1)
        self.bn4 = nn.BatchNorm3d(1)
        
        
        #self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.ELU()
        
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d):
                m.weight.data = nn.init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu')) 

    def forward(self, leaf_input):
     
        leaf_vector = self.deconv2(leaf_input)
        leaf_vector = self.bn2(leaf_vector)
        leaf_vector = self.tanh(leaf_vector)

        leaf_vector = self.deconv3(leaf_vector)
        leaf_vector = self.bn3(leaf_vector)
        leaf_vector = self.tanh(leaf_vector)       

        
        leaf_vector = self.deconv4(leaf_vector)
        leaf_vector = self.sigmoid(leaf_vector)
        leaf_vector = torch.clamp(leaf_vector, min=1e-7, max=1-1e-7)
        
        
        return leaf_vector

class ROctDecoder(nn.Module):
    def __init__(self, config):
        super(ROctDecoder, self).__init__()
        self.leaf_decoder = leafDecoder(feature_size = config.feature_size, leaf_size = config.leaf_code_size)
        
        self.node_decoder1 = NodeDecoder(feature_size = config.feature_size, hidden_size = config.hidden_size)
        self.node_decoder2 = NodeDecoder(feature_size = config.feature_size, hidden_size = config.hidden_size)
        self.node_decoder3 = NodeDecoder(feature_size = config.feature_size, hidden_size = config.hidden_size)
        self.node_decoder4 = NodeDecoder(feature_size = config.feature_size, hidden_size = config.hidden_size)
        self.node_decoder5 = NodeDecoder(feature_size = config.feature_size, hidden_size = config.hidden_size)
        
        self.sample_decoder = SampleDecoder(feature_size = config.feature_size, hidden_size = config.hidden_size)
        self.node_classifier = NodeClassifier(feature_size = config.feature_size, hidden_size = config.hidden_size)
        self.mseLoss = nn.MSELoss()  # pytorch's mean squared error loss
        self.creLoss = nn.CrossEntropyLoss()  # pytorch's cross entropy loss (NOTE: no softmax is needed before)

    def leafDecoder(self, feature):
        return self.leaf_decoder(feature)

    def NodeDecoder1(self, feature):
        return self.node_decoder1(feature)

    def NodeDecoder2(self, feature):
        return self.node_decoder2(feature)
    
    def NodeDecoder3(self, feature):
        return self.node_decoder3(feature)

    def NodeDecoder4(self, feature):
        return self.node_decoder4(feature)
    
    def NodeDecoder5(self, feature):
        return self.node_decoder5(feature)

    def sampleDecoder(self, feature):
        return self.sample_decoder(feature)

    def nodeClassifier(self, feature):
        return self.node_classifier(feature)

    def leafLossEstimator(self, leaf_feature, gt_leaf_feature):
        
        loss = torch.cat([torch.sum(-((gt.mul(5).mul(torch.log(b))).add((1-gt).mul(torch.log(1-b))))).mul(0.001) for b, gt in zip(leaf_feature, gt_leaf_feature)], 0)
        
#         print(gt_leaf_feature.size())
#         print(leaf_feature.size())
#         print('leafloss')
#         print(loss)
        return loss

#         return torch.cat([torch.sum(-(((gt.mul(3)-1).mul(40).mul(torch.log(b))).add((1-(gt.mul(3)-1)).mul(2).mul(torch.log(1-b))))).mul(0.0001) for b, gt in zip(leaf_feature, gt_leaf_feature)], 0)

    def classifyLossEstimator(self, label_vector, gt_label_vector):

        loss = torch.cat([self.creLoss(l.unsqueeze(0), gt).mul(10) for l, gt in zip(label_vector, gt_label_vector)], 0)
#         print('labelloss')
#         print(loss)
        return loss

    def vectorAdder(self, v1,v2,v3,v4,v5,v6,v7,v8):

            v1.add_(v2)
            v1.add_(v3)
            v1.add_(v4)
            v1.add_(v5)
            v1.add_(v6)
            v1.add_(v7)
            return v1.add_(v8)
        
    def vectorAdder2(self, v1,v2):
        return v1.add_(v2)
    
    def vectorZeros(self, v):
        return v.mul_(0)


def decode_structure_fold(fold, feature, tree):
    
    def decode_node_leaf(node, feature, l):
        if node.is_leaf():
            label = fold.add('nodeClassifier', feature)
            label_loss = fold.add('classifyLossEstimator', label, node.label)
            if node.is_empty_leaf():
                return fold.add('vectorZeros', label_loss),label_loss
            else:
                fea = fold.add('leafDecoder', feature)
                recon_loss = fold.add('leafLossEstimator', fea, node.fea)
                return recon_loss, label_loss
        
        elif node.is_expand():
            child1,child2,child3,child4,child5,child6,child7,child8 = fold.add('NodeDecoder'+str(l), feature).split(8)

            child_loss1, label_loss1 = decode_node_leaf(node.child[0], child1, l+1)
            child_loss2, label_loss2 = decode_node_leaf(node.child[1], child2, l+1)
            child_loss3, label_loss3 = decode_node_leaf(node.child[2], child3, l+1)
            child_loss4, label_loss4 = decode_node_leaf(node.child[3], child4, l+1)
            child_loss5, label_loss5 = decode_node_leaf(node.child[4], child5, l+1)
            child_loss6, label_loss6 = decode_node_leaf(node.child[5], child6, l+1)
            child_loss7, label_loss7 = decode_node_leaf(node.child[6], child7, l+1)
            child_loss8, label_loss8 = decode_node_leaf(node.child[7], child8, l+1)
            
#             child_loss1 = decode_node_leaf(node.child[0], child1, l+1)
#             child_loss2 = decode_node_leaf(node.child[1], child2, l+1)
#             child_loss3 = decode_node_leaf(node.child[2], child3, l+1)
#             child_loss4 = decode_node_leaf(node.child[3], child4, l+1)
#             child_loss5 = decode_node_leaf(node.child[4], child5, l+1)
#             child_loss6 = decode_node_leaf(node.child[5], child6, l+1)
#             child_loss7 = decode_node_leaf(node.child[6], child7, l+1)
#             child_loss8 = decode_node_leaf(node.child[7], child8, l+1)

            label = fold.add('nodeClassifier', feature)
            label_loss = fold.add('classifyLossEstimator', label, node.label)

            loss = fold.add('vectorAdder', child_loss1,child_loss2,child_loss3,child_loss4,child_loss5,child_loss6,child_loss7,child_loss8)
            
            child_label_loss = fold.add('vectorAdder', label_loss1,label_loss2,label_loss3,label_loss4,label_loss5,label_loss6,label_loss7,label_loss8)
            #loss = fold.add('vectorAdder2', loss, label_loss)
            return loss, fold.add('vectorAdder2', child_label_loss, label_loss)
            #return loss
            
    feature = fold.add('sampleDecoder', feature)
    loss,label_loss = decode_node_leaf(tree.root, feature,1)
    return loss,label_loss
