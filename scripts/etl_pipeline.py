import sys
from pathlib import Path

# Add parent folder to system path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from datetime import datetime
from sqlalchemy.orm import Session
from models.warehouse_schema import (
    DimAirline, DimAirport, DimTime, DimRoute, FactFlight, Base
)
from config.database import engine, SessionLocal


# ============================================================
#  STEP 1: CREATE DATABASE TABLES
# ============================================================
def create_tables():
    """Create all warehouse tables if they don’t already exist"""
    Base.metadata.create_all(bind=engine)
    print("✓ Database tables created successfully!")


# ============================================================
#  STEP 2: LOAD & CLEAN MULTIPLE DATASETS
# ============================================================
def load_and_clean_data():
    """Load, merge, and clean multiple flight datasets (Kaggle + generated formats)."""
    data_dir = Path("data")

    if not data_dir.exists():
        print(f"❌ Error: Data folder not found at {data_dir.absolute()}")
        return None

    all_csv_files = list(data_dir.glob("*.csv"))
    if not all_csv_files:
        print(f"❌ No CSV files found in {data_dir.absolute()}")
        return None

    print(f"✓ Found {len(all_csv_files)} CSV files:")
    for f in all_csv_files:
        print(f"   - {f.name}")

    # Helper function: ensure unique column names
    def make_unique_columns(columns):
        seen = {}
        unique = []
        for col in columns:
            if col in seen:
                seen[col] += 1
                unique.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                unique.append(col)
        return unique

    dfs = []
    for file in all_csv_files:
        try:
            df = pd.read_csv(file)
            if df.empty:
                print(f"⚠️  Skipping {file.name}: Empty file.")
                continue

            print(f"\nProcessing {file.name}...")
            print(f"Columns before cleaning: {list(df.columns)}")

            # ✅ Fix duplicate columns and reset index
            df.columns = make_unique_columns(df.columns)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            df.reset_index(drop=True, inplace=True)

            # Drop duplicate rows if any
            df = df[~df.index.duplicated(keep='first')]

            # -------------------------
            # COLUMN STANDARDIZATION
            # -------------------------
            column_mapping = {
                'Date': 'FLIGHT_DATE',
                'FlightDate': 'FLIGHT_DATE',
                'Airline': 'AIRLINE',
                'UniqueCarrier': 'AIRLINE',
                'Origin': 'ORIGIN_AIRPORT',
                'Org_Airport': 'ORIGIN_AIRPORT',
                'Dest': 'DESTINATION_AIRPORT',
                'Dest_Airport': 'DESTINATION_AIRPORT',
                'Destination': 'DESTINATION_AIRPORT',
                'DepDelay': 'DEPARTURE_DELAY',
                'ArrDelay': 'ARRIVAL_DELAY',
                'ScheduledDeparture': 'SCHEDULED_DEPARTURE',
                'ActualDeparture': 'ACTUAL_DEPARTURE',
                'ScheduledArrival': 'SCHEDULED_ARRIVAL',
                'ActualArrival': 'ACTUAL_ARRIVAL',
                'Distance': 'DISTANCE',
                'AirTime': 'AIR_TIME',
                'Cancelled': 'CANCELLED',
                'Diverted': 'DIVERTED',
            }

            df.rename(columns=column_mapping, inplace=True)

            # Ensure required columns exist
            required_cols = [
                'FLIGHT_DATE', 'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
                'DEPARTURE_DELAY', 'ARRIVAL_DELAY', 'CANCELLED', 'DIVERTED',
                'DISTANCE', 'AIR_TIME'
            ]
            for col in required_cols:
                if col not in df.columns:
                    df[col] = 0

            # Type conversions
            df['ARRIVAL_DELAY'] = pd.to_numeric(df['ARRIVAL_DELAY'], errors='coerce').fillna(0)
            df['DEPARTURE_DELAY'] = pd.to_numeric(df['DEPARTURE_DELAY'], errors='coerce').fillna(0)
            df['DISTANCE'] = pd.to_numeric(df['DISTANCE'], errors='coerce').fillna(0)
            df['AIR_TIME'] = pd.to_numeric(df['AIR_TIME'], errors='coerce').fillna(0)
            df['CANCELLED'] = pd.to_numeric(df['CANCELLED'], errors='coerce').fillna(0).astype(int)
            df['DIVERTED'] = pd.to_numeric(df['DIVERTED'], errors='coerce').fillna(0).astype(int)
            df['FLIGHT_DATE'] = pd.to_datetime(df['FLIGHT_DATE'], errors='coerce')

            # Drop missing key values
            df = df.dropna(subset=['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT'])

            # Add delay flag
            df['IS_DELAYED'] = (df['ARRIVAL_DELAY'] > 15).astype(int)

            if not df.empty:
                dfs.append(df)
            else:
                print(f"⚠️  Skipping {file.name}: No valid rows after cleaning.")

        except Exception as e:
            print(f"❌ Skipping {file.name} due to error: {e}")
            continue

    # ✅ Check before merge
    if not dfs:
        print("❌ Error: No valid datasets to merge. Please check your CSV files.")
        return None

    # ✅ Reset index and drop duplicates before concatenation
    for i, df in enumerate(dfs):
        df.reset_index(drop=True, inplace=True)
        df = df.loc[:, ~df.columns.duplicated()]
        dfs[i] = df

    # ✅ Safe concatenation (unique indices guaranteed)
    flights_df = pd.concat(dfs, axis=0, ignore_index=True)

    # Drop duplicates
    flights_df.drop_duplicates(
        subset=['AIRLINE', 'FLIGHT_DATE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT'],
        inplace=True
    )

    print("\n✅ Data merged and cleaned successfully!")
    print(f"Total records: {len(flights_df):,}")
    print(f"Unique airlines: {flights_df['AIRLINE'].nunique()}")
    print(f"Unique airports: {flights_df['ORIGIN_AIRPORT'].nunique() + flights_df['DESTINATION_AIRPORT'].nunique()}")
    print(f"Delayed flights: {flights_df['IS_DELAYED'].sum():,}")

    return flights_df


# ============================================================
#  STEP 3: POPULATE DIMENSIONS
# ============================================================
def populate_dimensions(db: Session, flights_df):
    print("\nPopulating dimension tables...")

    # Airlines
    for airline_code in flights_df['AIRLINE'].unique():
        if pd.notna(airline_code) and str(airline_code).strip() not in ["", "0", "nan"]:
            existing = db.query(DimAirline).filter_by(airline_code=airline_code[:3]).first()
            if not existing:
                db.add(DimAirline(airline_code=airline_code[:3]))
    db.commit()
    print("✓ dim_airline populated.")

    # Airports
    airports = pd.unique(flights_df[['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']].values.ravel())
    for code in airports:
        if pd.notna(code) and str(code).strip() not in ["", "0", "nan"]:
            existing = db.query(DimAirport).filter_by(airport_code=code[:3]).first()
            if not existing:
                db.add(DimAirport(airport_code=code[:3]))
    db.commit()
    print("✓ dim_airport populated.")

    # Time
    flights_df['FLIGHT_DATE'] = pd.to_datetime(flights_df['FLIGHT_DATE'], errors='coerce')
    for date in flights_df['FLIGHT_DATE'].dropna().unique():
        d = pd.Timestamp(date)
        exists = db.query(DimTime).filter_by(flight_date=d.date()).first()
        if not exists:
            db.add(DimTime(
                flight_date=d.date(),
                year=d.year,
                month=d.month,
                day=d.day,
                day_of_week=d.dayofweek,
                quarter=(d.month - 1)//3 + 1,
                week_of_year=d.isocalendar()[1],
                is_weekend=1 if d.dayofweek >= 5 else 0
            ))
    db.commit()
    print("✓ dim_time populated.")

    # Routes
    routes = flights_df[['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']].drop_duplicates()
    for _, row in routes.iterrows():
        origin = db.query(DimAirport).filter_by(airport_code=row['ORIGIN_AIRPORT'][:3]).first()
        dest = db.query(DimAirport).filter_by(airport_code=row['DESTINATION_AIRPORT'][:3]).first()
        if origin and dest:
            route_code = f"{row['ORIGIN_AIRPORT'][:3]}-{row['DESTINATION_AIRPORT'][:3]}"
            exists = db.query(DimRoute).filter_by(route_code=route_code).first()
            if not exists:
                db.add(DimRoute(
                    origin_airport_id=origin.airport_id,
                    dest_airport_id=dest.airport_id,
                    route_code=route_code
                ))
    db.commit()
    print("✓ dim_route populated.")


# ============================================================
#  STEP 4: POPULATE FACT TABLE
# ============================================================
def populate_facts(db: Session, flights_df):
    print("\nPopulating fact_flight table...")

    airlines = {a.airline_code: a.airline_id for a in db.query(DimAirline).all()}
    airports = {a.airport_code: a.airport_id for a in db.query(DimAirport).all()}
    times = {t.flight_date: t.time_id for t in db.query(DimTime).all()}

    loaded_count = 0
    for _, row in flights_df.iterrows():
        try:
            date = pd.Timestamp(row['FLIGHT_DATE']).date()
            origin = str(row['ORIGIN_AIRPORT'])[:3]
            dest = str(row['DESTINATION_AIRPORT'])[:3]
            airline = str(row['AIRLINE'])[:3]

            route_code = f"{origin}-{dest}"
            route = db.query(DimRoute).filter_by(route_code=route_code).first()

            if route and all([origin, dest, airline]) and date in times:
                db.add(FactFlight(
                    airline_id=airlines.get(airline),
                    origin_airport_id=airports.get(origin),
                    dest_airport_id=airports.get(dest),
                    route_id=route.route_id,
                    time_id=times.get(date),
                    departure_delay=float(row.get('DEPARTURE_DELAY', 0)),
                    arrival_delay=float(row.get('ARRIVAL_DELAY', 0)),
                    is_delayed=int(row.get('IS_DELAYED', 0)),
                    is_cancelled=int(row.get('CANCELLED', 0)),
                    is_diverted=int(row.get('DIVERTED', 0)),
                    air_time=float(row.get('AIR_TIME', 0)),
                    distance=float(row.get('DISTANCE', 0))
                ))
                loaded_count += 1
        except Exception:
            continue

    db.commit()
    print(f"✓ fact_flight populated with {loaded_count:,} records.")


# ============================================================
#  STEP 5: RUN COMPLETE ETL
# ============================================================
def run_etl():
    print("\n" + "="*60)
    print("🚀 STARTING ETL PIPELINE")
    print("="*60)

    create_tables()
    flights_df = load_and_clean_data()
    if flights_df is None:
        return

    db = SessionLocal()
    try:
        populate_dimensions(db, flights_df)
        populate_facts(db, flights_df)
        print("\n" + "="*60)
        print("🎉 ETL PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60 + "\n")
    except Exception as e:
        db.rollback()
        print(f"❌ ETL Error: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    run_etl()
