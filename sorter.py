import pandas as pd

class Sorter:
    csv = None

    def load(self, file_path):
        # note 2022: for chunk in read_csv(..., chunksize=x) -> chunk.itertuples()
        self.csv = pd.read_csv(file_path)

    # returns [[record_id, yield_per_acre]]
    def sort_by_yield(self, n) -> [[int, int]]:
        sorted = [[0, 0] for _ in range(n)]
        for record in self.csv.itertuples():
            for sorted_record in sorted:
                if sorted_record[1] < record.yieldPerAcre:
                    sorted_record[0] = record.Index
                    sorted_record[1] = record.yieldPerAcre
                    break
        return sorted

    def calculate_naive_feature_weight(self, feature_idx, top_n):
        feature_variant_map = {}
        for record in self.csv.itertuples():
            feature_variant = record[feature_idx]
            if pd.isna(feature_variant):
                continue
            
            # Simplified list management
            if feature_variant not in feature_variant_map:
                feature_variant_map[feature_variant] = []
            feature_variant_map[feature_variant].append(record.yieldPerAcre)

        # Calculate averages
        for variant_name, variant_list in feature_variant_map.items():
            variant_avg = sum(variant_list) / len(variant_list)
            feature_variant_map[variant_name] = variant_avg

        # Sort by average yield (value) descending and take the top N
        sorted_variants = sorted(
            feature_variant_map.items(), 
            key=lambda item: item[1], 
            reverse=True
        )[:top_n]

        return dict(sorted_variants)
