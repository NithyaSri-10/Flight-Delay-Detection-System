import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models.warehouse_schema import Base, DimAirline, DimAirport, DimTime, FactFlight
from services.olap_queries import OLAPAnalytics
from datetime import datetime, date

@pytest.fixture
def test_db():
    """Create test database"""
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()

@pytest.fixture
def sample_warehouse_data(test_db):
    """Create sample data in warehouse"""
    # Add airlines
    airline = DimAirline(airline_code='AA', airline_name='American Airlines')
    test_db.add(airline)
    
    # Add airports
    airport1 = DimAirport(airport_code='ATL', airport_name='Hartsfield-Jackson', city='Atlanta', state='GA')
    airport2 = DimAirport(airport_code='LAX', airport_name='Los Angeles', city='Los Angeles', state='CA')
    test_db.add(airport1)
    test_db.add(airport2)
    
    # Add time dimension
    time_dim = DimTime(
        flight_date=date(2024, 1, 1),
        year=2024, month=1, day=1,
        day_of_week=0, quarter=1, week_of_year=1,
        is_weekend=0
    )
    test_db.add(time_dim)
    test_db.commit()
    
    # Add fact records
    for i in range(10):
        fact = FactFlight(
            airline_id=airline.airline_id,
            origin_airport_id=airport1.airport_id,
            dest_airport_id=airport2.airport_id,
            time_id=time_dim.time_id,
            departure_delay=float(i * 5),
            arrival_delay=float(i * 5 + 10),
            is_delayed=1 if i > 5 else 0,
            is_cancelled=0,
            is_diverted=0,
            distance=2100
        )
        test_db.add(fact)
    
    test_db.commit()
    return test_db

class TestOLAPQueries:
    """Test OLAP analytics queries"""
    
    def test_overall_statistics(self, sample_warehouse_data):
        """Test overall statistics query"""
        stats = OLAPAnalytics.get_overall_statistics(sample_warehouse_data)
        
        assert stats is not None
        assert 'total_flights' in stats
        assert stats['total_flights'] == 10
    
    def test_delay_by_airline(self, sample_warehouse_data):
        """Test delay by airline query"""
        results = OLAPAnalytics.get_delay_by_airline(sample_warehouse_data)
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert 'airline_code' in results[0]
        assert 'delay_percentage' in results[0]
    
    def test_delay_by_airport(self, sample_warehouse_data):
        """Test delay by airport query"""
        results = OLAPAnalytics.get_delay_by_airport(sample_warehouse_data, 'origin')
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert 'airport_code' in results[0]
    
    def test_delay_by_route(self, sample_warehouse_data):
        """Test delay by route query"""
        results = OLAPAnalytics.get_delay_by_route(sample_warehouse_data, limit=5)
        
        assert isinstance(results, list)
        assert 'route_code' in results[0]
