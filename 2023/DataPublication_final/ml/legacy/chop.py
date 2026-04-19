import pandas as pd

# Load the giant master file
master_df = pd.read_csv('../GroundTruth/train_HIPS_HYBRIDS_2023_V2.3.csv')

# Example A: Partition by Location
lincoln_df = master_df[master_df['location'] == 'Lincoln']
lincoln_df.to_csv('lincoln_plan.csv', index=False)
print(f"Saved Lincoln data: {len(lincoln_df)} rows")

# Example B: Partition by specific Experiment ID
exp1_df = master_df[master_df['experiment'] == 'Exp_101']
exp1_df.to_csv('exp101_plan.csv', index=False)