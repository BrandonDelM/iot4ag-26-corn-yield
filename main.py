import sorter as sorter_module

def main():
    file_path = "2023/DataPublication_final/GroundTruth/train_HIPS_HYBRIDS_2023_V2.3.csv"
    sorter = sorter_module.Sorter()
    sorter.load(file_path)
    sorted = sorter.sort_by_yield(5)
    print(sorted)

if __name__ == "__main__":
    main()
