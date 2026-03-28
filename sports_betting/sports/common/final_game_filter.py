import pandas as pd
from datetime import datetime, timedelta
import pytz

ET = pytz.timezone("America/New_York")


def filter_predictions_today(df):
    print("ENTERING FUNCTION: filter_predictions_today", len(df))
    if "metadata" not in df.columns:
        return df

    start = datetime.now(ET).replace(hour=4, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)

    def keep_row(row):
        try:
            meta = row["metadata"]

            if isinstance(meta, str):
                import json

                meta = json.loads(meta)

            commence = pd.to_datetime(meta["commence_time"], utc=True)
            commence_et = commence.tz_convert(ET)

            return start <= commence_et <= end

        except Exception:
            return True

    out = df[df.apply(keep_row, axis=1)]
    print("EXIT FUNCTION: filter_predictions_today", len(out))
    return out
