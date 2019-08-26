from collections import deque

import torch
import torch.nn as nn
from torchvision import models

from models.mfb_fusion import MFBFusion
from options import GlobalOption
from .base_model import BaseModel


class TreeNode(BaseModel):
    def __init__(self, image_encoder_option, node):
        super(TreeNode, self).__init__()
        self.node = node
        self.conv = nn.Conv2d(image_encoder_option.in_channel_of_depth[node.depth],
                              image_encoder_option.out_channel_of_depth[node.depth],
                              image_encoder_option.kernel_size_of_depth[node.depth]).to(GlobalOption.device)

    def forward(self, image):
        return self.conv(image)


class ImageEncoder(BaseModel):
    def __init__(self, image_encoder_option):
        super(ImageEncoder, self).__init__()
        self.cnn = models.resnet18(pretrained=True).to(GlobalOption.device)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-5]).to(GlobalOption.device)
        self.taxonomy_tree = image_encoder_option.taxonomy_tree
        self.cnn_taxonomy_nodes = [TreeNode(image_encoder_option, node) for node in self.taxonomy_tree.taxonomy_nodes]
        self.cnn_attribute_nodes = [TreeNode(image_encoder_option, node) for node in self.taxonomy_tree.attribute_nodes]
        self.cnn_attribute_item_nodes = [TreeNode(image_encoder_option, node) for node in
                                         self.taxonomy_tree.attribute_item_nodes]
        self.num_taxonomy_nodes = len(self.taxonomy_tree.taxonomy_nodes)
        self.num_last_layer_taxonomy_nodes = len(self.taxonomy_tree.last_layer_taxonomy_nodes)
        self.num_attributes = len(self.cnn_attribute_nodes)
        self.softmax = nn.Softmax(dim=0)
        self.mfb = MFBFusion(image_encoder_option.mfb_fusion).to(GlobalOption.device)
        item_nums = torch.zeros(len(self.cnn_attribute_nodes), dtype=torch.long)
        for node in self.cnn_attribute_item_nodes:
            item_nums[node.node.parent] += 1
        item_nums = torch.cumsum(item_nums, 0)
        self.item_interval = []
        for i, s in enumerate(item_nums):
            self.item_interval.append((item_nums[i - 1] if i > 0 else 0, s))

    def bfs(self, cnned):
        queue = deque()
        queue.append(0)
        while len(queue) > 0:
            u = queue.pop()
            for child_idx in self.cnn_taxonomy_nodes[u].node.children:
                cnned[child_idx] = self.cnn_taxonomy_nodes[child_idx](cnned[u]).to(GlobalOption.device)
                queue.append(child_idx)

    def forward(self, text, image, prod_taxonomy, prod_attributes):
        # text: (batch_size, hidden_size)
        # image: (batch_size, 3, image_size, image_size)
        # prod_taxonomy: (batch_size, )
        # prod_attributes: (batch_size, num_attributes)

        batch_size = text.size(0)

        node_x = [None for _ in range(len(self.cnn_taxonomy_nodes))]
        node_x[0] = self.cnn(image)
        self.bfs(node_x)
        # node_x: (num_taxonomy_nodes, batch, C, H, W)
        node_x = torch.stack(node_x[-self.num_last_layer_taxonomy_nodes:]).to(GlobalOption.device)
        # node_x: (num_last_layer_taxonomy_nodes, batch, C, H, W)
        taxonomies = prod_taxonomy - (self.num_taxonomy_nodes - self.num_last_layer_taxonomy_nodes)
        # taxonomies: (batch, )
        taxonomy_one_hot = torch.zeros(batch_size, self.num_last_layer_taxonomy_nodes, dtype=torch.long).to(GlobalOption.device)
        taxonomy_one_hot.scatter_(1, taxonomies.view(-1, 1), 1)
        taxonomy_one_hot.transpose_(0, 1)
        # taxonomy_one_hot (num_last_layer_taxonomy_nodes, batch)
        taxonomy_one_hot = taxonomy_one_hot.float()
        taxonomy_one_hot.unsqueeze_(2).unsqueeze_(3).unsqueeze_(4)
        # taxonomy_one_hot (num_last_layer_taxonomy_nodes, batch, 1, 1, 1)
        node_x = node_x.mul(taxonomy_one_hot)
        # (num_last_layer_taxonomy_nodes, batch, C, H, W)
        node_x = node_x.sum(0)
        # node_x: (batch, C, H, W)
        attr_x = [attr_cnn(node_x) for attr_cnn in self.cnn_attribute_nodes]
        # attr_x: (num_attrs, batch, C, H, W)
        leaf_x = torch.stack([leaf_cnn(attr_x[leaf_cnn.node.parent]) for leaf_cnn in self.cnn_attribute_item_nodes]).to(GlobalOption.device)
        # leaf_x: (num_items, batch, C, H, W)

        # taxonomies: (batch, )
        attribute_one_hot = torch.zeros(batch_size, len(self.cnn_attribute_item_nodes), dtype=torch.long).to(GlobalOption.device)
        attribute_one_hot.scatter_(1, prod_attributes, 1)
        attribute_one_hot.transpose_(0, 1)
        # (num_items, batch)
        attribute_one_hot = attribute_one_hot.float()
        attribute_one_hot.unsqueeze_(2).unsqueeze_(3).unsqueeze_(4)
        # (num_items, batch, 1, 1, 1)
        leaf_x = leaf_x.mul(attribute_one_hot)
        # (num_items, batch, C, H, W)
        leaf_x = torch.stack([torch.sum(leaf_x[i: j], 0) for i, j in self.item_interval]).view(len(self.item_interval), batch_size, -1)
        # leaf_x: (num_attrs, batch, C * H * W)

        scores = torch.stack([self.mfb(text, image) for image in leaf_x])
        # scores: (num_attrs, batch)

        scores = self.softmax(scores).unsqueeze(2)
        # scores: (num_attrs, batch, 1)
        attr_x = torch.mul(leaf_x, scores)
        # attr_x: (num_attrs, batch, C * H * W)
        attr_x = torch.sum(attr_x, 0)
        # attr_x: (batch, C * H * W)
        return attr_x
