pub enum TreeNode {
    Split {
        feature_index: usize,
        threshold: f32,
        left_child: Box<TreeNode>,
        right_child: Box<TreeNode>,
    },
    Leaf {
        value: f32,
    },
}

pub struct Tree {
    root: Box<TreeNode>,
}

impl Tree {
    pub fn new(root: Box<TreeNode>) -> Self {
        Self { root }
    }

    pub fn predict(&self, features: &[f32]) -> f32 {
        Self::predict_recursive(&self.root, features)
    }

    fn predict_recursive(node: &TreeNode, features: &[f32]) -> f32 {
        match node {
            TreeNode::Leaf { value } => *value,
            TreeNode::Split {
                feature_index,
                threshold,
                left_child,
                right_child,
            } => {
                let feature_value = features[*feature_index];

                if feature_value < *threshold {
                    Self::predict_recursive(left_child, features)
                } else {
                    Self::predict_recursive(right_child, features)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*; // Import Tree and TreeNode from parent module

    #[test]
    fn test_simple_tree_prediction() {
        let left_leaf = TreeNode::Leaf { value: 10.0 };
        let right_leaf = TreeNode::Leaf { value: 20.0 };

        let root = TreeNode::Split {
            feature_index: 0,
            threshold: 5.0,
            left_child: Box::new(left_leaf),
            right_child: Box::new(right_leaf),
        };

        let tree = Tree::new(Box::new(root));

        assert_eq!(tree.predict(&[3.0]), 10.0);
    }
}
