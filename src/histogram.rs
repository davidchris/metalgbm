pub struct Histogram {
    bins: Vec<f32>,
    gradients: Vec<f32>,
    hessians: Vec<f32>, // first derivative of loss function
}

impl Histogram {
    pub fn from_feature(feature_values: &[f32], max_bins: usize) -> Self {
        // this functions defines the bins of the histogram
        let mut sorted_values: Vec<f32> = feature_values.iter().copied().collect();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted_values.dedup();

        let n_unique = sorted_values.len();

        if n_unique == 0 {
            return Self {
                bins: vec![],
                gradients: vec![],
                hessians: vec![],
            };
        }

        if n_unique == 1 {
            return Self {
                bins: vec![sorted_values[0]],
                gradients: vec![0.0],
                hessians: vec![0.0],
            };
        }

        let num_bins = max_bins.min(n_unique - 1);

        let mut bins = Vec::new();
        for i in 0..=num_bins {
            let idx = (i * (n_unique - 1)) / num_bins;
            bins.push(sorted_values[idx]);
        }

        let gradients = vec![0.0; num_bins];
        let hessians = vec![0.0; num_bins];

        Self {
            bins,
            gradients,
            hessians,
        }
    }

    pub fn accumulate(&mut self, feature_values: &[f32], gradients: &[f32], hessians: &[f32]) {
        // For each sample:
        //  1. Find which bin the feature value falls into
        //  2. Add the sample's gradient to that bin's gradient sum
        //  3. Add the sample's hessian to that bin's hessian sum
        //
        //  Algorithm:
        //      - For sample i with feature value `feature_values[i]
        //          - Find bin index: binary search through self.bins to find where the value falls
        //          - Accumulate: self.gradients[bin_idx] += gradients[i]
        //          - Accumulate: self.hessians[bin_idx] += hessians[i]

        for i in 0..feature_values.len() {
            let bin_idx = self.search_bin_index(&feature_values[i]);
            self.gradients[bin_idx] += gradients[i];
            self.hessians[bin_idx] += hessians[i];
        }
    }

    fn search_bin_index(&self, feature_value: &f32) -> usize {
        // Find the first bin boundary that is strictly greater than feature_value
        let idx = self
            .bins
            .partition_point(|&boundary| boundary <= *feature_value);

        if idx == 0 {
            0
        } else {
            (idx - 1).min(self.gradients.len() - 1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_from_feature_normal_case() {
        let feature_vec = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let hist = Histogram::from_feature(&feature_vec, 4);

        assert_eq!(hist.bins.len(), 5);
        assert_eq!(hist.bins, [0.0, 2.0, 4.0, 6.0, 9.0]);
        assert_eq!(hist.gradients.len(), 4);
        assert_eq!(hist.hessians.len(), 4);
    }

    #[test]
    fn test_from_feature_fewer_unique_values() {
        // Only 3 unique values, but requesting 10 bins
        let feature_vec = vec![1.0, 5.0, 10.0, 1.0, 5.0, 10.0];
        let hist = Histogram::from_feature(&feature_vec, 10);

        // With 3 unique values, we only create 2 bins (3 boundaries)
        assert_eq!(hist.bins.len(), 3);
        assert_eq!(hist.bins, [1.0, 5.0, 10.0]);
        assert_eq!(hist.gradients.len(), 2);
        assert_eq!(hist.hessians.len(), 2);
    }

    #[test]
    fn test_from_feature_single_value() {
        let feature_vec = vec![42.0, 42.0, 42.0, 42.0, 42.0];
        let hist = Histogram::from_feature(&feature_vec, 5);

        // Should trigger the n_unique == 1 case
        assert_eq!(hist.bins.len(), 1);
        assert_eq!(hist.bins[0], 42.0);
        assert_eq!(hist.gradients.len(), 1);
        assert_eq!(hist.hessians.len(), 1);
    }

    #[test]
    fn test_from_feature_empty() {
        let feature_vec: Vec<f32> = vec![];
        let hist = Histogram::from_feature(&feature_vec, 5);

        // Should trigger the n_unique == 0 case
        assert_eq!(hist.bins.len(), 0);
        assert_eq!(hist.gradients.len(), 0);
        assert_eq!(hist.hessians.len(), 0);
    }

    #[test]
    fn test_search_bin_index() {
        let feature_values = vec![0.0, 2.0, 4.0, 6.0, 9.0];
        let hist = Histogram::from_feature(&feature_values, 4);

        // test values that fall cleanly in bins
        assert_eq!(hist.search_bin_index(&1.0), 0);
        assert_eq!(hist.search_bin_index(&3.0), 1);
        assert_eq!(hist.search_bin_index(&5.0), 2);
        assert_eq!(hist.search_bin_index(&7.0), 3);

        assert_eq!(hist.search_bin_index(&2.0), 1);
        assert_eq!(hist.search_bin_index(&4.0), 2);

        assert_eq!(hist.search_bin_index(&0.0), 0);
        assert_eq!(hist.search_bin_index(&9.0), 3);
        assert_eq!(hist.search_bin_index(&10.0), 3);
    }

    #[test]
    fn test_accumulate() {
        let feature_values = vec![1.0, 2.0, 3.0, 5.0, 7.0];
        let mut hist = Histogram::from_feature(&feature_values, 2);

        let gradients = vec![-0.5, 0.3, -0.2, 0.4, 0.1];
        let hessians = vec![1.0, 1.2, 0.9, 1.1, 1.0];

        hist.accumulate(&feature_values, &gradients, &hessians);

        assert_abs_diff_eq!(hist.gradients[0], -0.2, epsilon = 1e-6);
        assert_abs_diff_eq!(hist.gradients[1], 0.3, epsilon = 1e-6);

        assert_abs_diff_eq!(hist.hessians[0], 2.2, epsilon = 1e-6);
        assert_abs_diff_eq!(hist.hessians[1], 3.0, epsilon = 1e-6);
    }
}
