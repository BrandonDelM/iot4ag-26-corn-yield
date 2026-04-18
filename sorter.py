import pandas as pd

class Sorter:
    csv = None

    def load(self, file_path):
        # note 2022: for chunk in read_csv(..., chunksize=x) -> chunk.itertuples()
        self.csv = pd.read_csv(file_path, usecols=["yieldPerAcre"])

    def sort_by_yield(self, n):
        # sort for top n yield records.
        # if record.yield > lowest_sorted_yield
        #   search sorted for lowest
        #   replace lowest found with record.yield
        sorted = [0] * n
        lowest_sorted_yield = 0
        for record in self.csv.itertuples():
            if record.yieldPerAcre > lowest_sorted_yield:
                for i, v in enumerate(sorted):
                    if v < record.yieldPerAcre:
                        sorted[i] = record.yieldPerAcre
                        break
        return sorted
