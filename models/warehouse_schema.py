from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Date, Time
from sqlalchemy.orm import relationship
from config.database import Base
from datetime import datetime

# DIMENSION TABLES

class DimAirline(Base):
    __tablename__ = "dim_airline"
    
    airline_id = Column(Integer, primary_key=True)
    airline_code = Column(String(3), unique=True, nullable=False)
    airline_name = Column(String(100))
    
    flights = relationship("FactFlight", back_populates="airline")

class DimAirport(Base):
    __tablename__ = "dim_airport"
    
    airport_id = Column(Integer, primary_key=True)
    airport_code = Column(String(3), unique=True, nullable=False)
    airport_name = Column(String(100))
    city = Column(String(50))
    state = Column(String(2))
    
    origin_flights = relationship("FactFlight", foreign_keys="FactFlight.origin_airport_id", back_populates="origin_airport")
    dest_flights = relationship("FactFlight", foreign_keys="FactFlight.dest_airport_id", back_populates="dest_airport")

class DimTime(Base):
    __tablename__ = "dim_time"
    
    time_id = Column(Integer, primary_key=True)
    flight_date = Column(Date, nullable=False)
    year = Column(Integer)
    month = Column(Integer)
    day = Column(Integer)
    day_of_week = Column(Integer)  # 0=Monday, 6=Sunday
    quarter = Column(Integer)
    week_of_year = Column(Integer)
    is_weekend = Column(Integer)
    
    flights = relationship("FactFlight", back_populates="time_dim")

class DimRoute(Base):
    __tablename__ = "dim_route"
    
    route_id = Column(Integer, primary_key=True)
    origin_airport_id = Column(Integer, ForeignKey("dim_airport.airport_id"))
    dest_airport_id = Column(Integer, ForeignKey("dim_airport.airport_id"))
    route_code = Column(String(10), unique=True)
    
    flights = relationship("FactFlight", back_populates="route")

# FACT TABLE

class FactFlight(Base):
    __tablename__ = "fact_flight"
    
    flight_id = Column(Integer, primary_key=True)
    airline_id = Column(Integer, ForeignKey("dim_airline.airline_id"))
    origin_airport_id = Column(Integer, ForeignKey("dim_airport.airport_id"))
    dest_airport_id = Column(Integer, ForeignKey("dim_airport.airport_id"))
    route_id = Column(Integer, ForeignKey("dim_route.route_id"))
    time_id = Column(Integer, ForeignKey("dim_time.time_id"))
    
    scheduled_departure = Column(Time)
    scheduled_arrival = Column(Time)
    actual_departure = Column(Time)
    actual_arrival = Column(Time)
    
    departure_delay = Column(Float)  # in minutes
    arrival_delay = Column(Float)    # in minutes
    is_delayed = Column(Integer)     # 1 if arrival_delay > 15, else 0
    is_cancelled = Column(Integer)
    is_diverted = Column(Integer)
    
    air_time = Column(Float)         # in minutes
    distance = Column(Float)         # in miles
    
    # Relationships
    airline = relationship("DimAirline", back_populates="flights")
    origin_airport = relationship("DimAirport", foreign_keys=[origin_airport_id], back_populates="origin_flights")
    dest_airport = relationship("DimAirport", foreign_keys=[dest_airport_id], back_populates="dest_flights")
    route = relationship("DimRoute", back_populates="flights")
    time_dim = relationship("DimTime", back_populates="flights")
    
    created_at = Column(DateTime, default=datetime.utcnow)
