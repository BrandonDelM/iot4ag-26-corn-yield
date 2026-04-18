import pandas as pd

class Sorter:
    csv = None

    def load(self, file_path):
        # note 2022: for chunk in read_csv(..., chunksize=x) -> chunk.itertuples()
        self.csv = pd.read_csv(file_path, usecols=["yieldPerAcre"])

    # returns [[record_id, yield_per_acre]]
    def sort_by_yield(self, n) -> [[int, int]]:
        sorted = [[0, 0] for _ in range(n)]
        for record in self.csv.itertuples():
            for sorted_record in sorted:
                if sorted_record[1] < record.yieldPerAcre:
                    sorted_record[0] = record.Index
                    sorted_record[1] = record.yieldPerAcre
                    break;
        return sorted
