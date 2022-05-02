def callback(schedule_csv, schedule_idx):
    import pandas as pd
    from datetime import datetime

    schedule = pd.read_csv(schedule_csv, index_col=0)
    schedule.at[schedule_idx, 'ended'] = datetime.now().strftime("%Y_%m_%d_%H_%M")
    schedule.to_csv(schedule_csv)
