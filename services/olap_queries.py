from sqlalchemy import func, and_
from sqlalchemy.orm import Session
from models.warehouse_schema import (
    FactFlight, DimAirline, DimAirport, DimTime, DimRoute
)

class OLAPAnalytics:
    """OLAP queries for analytical operations"""
    
    @staticmethod
    def get_delay_by_airline(db: Session):
        """Get average delay metrics by airline"""
        results = db.query(
            DimAirline.airline_code,
            DimAirline.airline_name,
            func.count(FactFlight.flight_id).label('total_flights'),
            func.sum(FactFlight.is_delayed).label('delayed_flights'),
            func.avg(FactFlight.arrival_delay).label('avg_arrival_delay'),
            func.avg(FactFlight.departure_delay).label('avg_departure_delay'),
            (func.sum(FactFlight.is_delayed) * 100.0 / func.count(FactFlight.flight_id)).label('delay_percentage')
        ).join(FactFlight).group_by(
            DimAirline.airline_id, DimAirline.airline_code, DimAirline.airline_name
        ).order_by(func.avg(FactFlight.arrival_delay).desc()).all()
        
        return [dict(zip(['airline_code', 'airline_name', 'total_flights', 'delayed_flights', 
                         'avg_arrival_delay', 'avg_departure_delay', 'delay_percentage'], r)) for r in results]
    
    @staticmethod
    def get_delay_by_airport(db: Session, airport_type='origin'):
        """Get delay metrics by airport (origin or destination)"""
        if airport_type == 'origin':
            airport_col = FactFlight.origin_airport_id
            airport_rel = FactFlight.origin_airport
        else:
            airport_col = FactFlight.dest_airport_id
            airport_rel = FactFlight.dest_airport
        
        results = db.query(
            DimAirport.airport_code,
            DimAirport.airport_name,
            DimAirport.city,
            func.count(FactFlight.flight_id).label('total_flights'),
            func.sum(FactFlight.is_delayed).label('delayed_flights'),
            func.avg(FactFlight.arrival_delay).label('avg_arrival_delay'),
            (func.sum(FactFlight.is_delayed) * 100.0 / func.count(FactFlight.flight_id)).label('delay_percentage')
        ).join(FactFlight, airport_col == DimAirport.airport_id).group_by(
            DimAirport.airport_id, DimAirport.airport_code, DimAirport.airport_name, DimAirport.city
        ).order_by(func.avg(FactFlight.arrival_delay).desc()).all()
        
        return [dict(zip(['airport_code', 'airport_name', 'city', 'total_flights', 'delayed_flights',
                         'avg_arrival_delay', 'delay_percentage'], r)) for r in results]
    
    @staticmethod
    def get_delay_by_route(db: Session, limit=20):
        """Get delay metrics by route"""
        results = db.query(
            DimRoute.route_code,
            func.count(FactFlight.flight_id).label('total_flights'),
            func.sum(FactFlight.is_delayed).label('delayed_flights'),
            func.avg(FactFlight.arrival_delay).label('avg_arrival_delay'),
            (func.sum(FactFlight.is_delayed) * 100.0 / func.count(FactFlight.flight_id)).label('delay_percentage')
        ).join(FactFlight).group_by(
            DimRoute.route_id, DimRoute.route_code
        ).order_by(func.avg(FactFlight.arrival_delay).desc()).limit(limit).all()
        
        return [dict(zip(['route_code', 'total_flights', 'delayed_flights', 'avg_arrival_delay', 'delay_percentage'], r)) for r in results]
    
    @staticmethod
    def get_delay_by_hour(db: Session):
        """Get delay metrics by hour of day"""
        results = db.query(
            func.extract('hour', FactFlight.scheduled_departure).label('hour'),
            func.count(FactFlight.flight_id).label('total_flights'),
            func.sum(FactFlight.is_delayed).label('delayed_flights'),
            func.avg(FactFlight.arrival_delay).label('avg_arrival_delay'),
            (func.sum(FactFlight.is_delayed) * 100.0 / func.count(FactFlight.flight_id)).label('delay_percentage')
        ).group_by(
            func.extract('hour', FactFlight.scheduled_departure)
        ).order_by('hour').all()
        
        return [dict(zip(['hour', 'total_flights', 'delayed_flights', 'avg_arrival_delay', 'delay_percentage'], r)) for r in results]
    
    @staticmethod
    def get_delay_by_day_of_week(db: Session):
        """Get delay metrics by day of week"""
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        results = db.query(
            DimTime.day_of_week,
            func.count(FactFlight.flight_id).label('total_flights'),
            func.sum(FactFlight.is_delayed).label('delayed_flights'),
            func.avg(FactFlight.arrival_delay).label('avg_arrival_delay'),
            (func.sum(FactFlight.is_delayed) * 100.0 / func.count(FactFlight.flight_id)).label('delay_percentage')
        ).join(FactFlight).group_by(
            DimTime.day_of_week
        ).order_by(DimTime.day_of_week).all()
        
        return [dict(zip(['day_of_week', 'day_name', 'total_flights', 'delayed_flights', 'avg_arrival_delay', 'delay_percentage'], 
                        (r[0], day_names[int(r[0])], r[1], r[2], r[3], r[4]))) for r in results]
    
    @staticmethod
    def get_delay_by_month(db: Session):
        """Get delay metrics by month"""
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        results = db.query(
            DimTime.month,
            func.count(FactFlight.flight_id).label('total_flights'),
            func.sum(FactFlight.is_delayed).label('delayed_flights'),
            func.avg(FactFlight.arrival_delay).label('avg_arrival_delay'),
            (func.sum(FactFlight.is_delayed) * 100.0 / func.count(FactFlight.flight_id)).label('delay_percentage')
        ).join(FactFlight).group_by(
            DimTime.month
        ).order_by(DimTime.month).all()
        
        return [dict(zip(['month', 'month_name', 'total_flights', 'delayed_flights', 'avg_arrival_delay', 'delay_percentage'],
                        (r[0], month_names[int(r[0])-1], r[1], r[2], r[3], r[4]))) for r in results]
    
    @staticmethod
    def get_overall_statistics(db: Session):
        """Get overall delay statistics"""
        result = db.query(
            func.count(FactFlight.flight_id).label('total_flights'),
            func.sum(FactFlight.is_delayed).label('delayed_flights'),
            func.avg(FactFlight.arrival_delay).label('avg_arrival_delay'),
            func.avg(FactFlight.departure_delay).label('avg_departure_delay'),
            func.max(FactFlight.arrival_delay).label('max_arrival_delay'),
            func.min(FactFlight.arrival_delay).label('min_arrival_delay'),
            (func.sum(FactFlight.is_delayed) * 100.0 / func.count(FactFlight.flight_id)).label('delay_percentage')
        ).first()
        
        return dict(zip(['total_flights', 'delayed_flights', 'avg_arrival_delay', 'avg_departure_delay',
                        'max_arrival_delay', 'min_arrival_delay', 'delay_percentage'], result))
    
    @staticmethod
    def drill_down_route_details(db: Session, route_code: str):
        """Drill-down: Get detailed metrics for a specific route"""
        results = db.query(
            DimTime.month,
            func.count(FactFlight.flight_id).label('total_flights'),
            func.sum(FactFlight.is_delayed).label('delayed_flights'),
            func.avg(FactFlight.arrival_delay).label('avg_arrival_delay'),
            (func.sum(FactFlight.is_delayed) * 100.0 / func.count(FactFlight.flight_id)).label('delay_percentage')
        ).join(FactFlight).join(DimRoute).filter(
            DimRoute.route_code == route_code
        ).group_by(DimTime.month).order_by(DimTime.month).all()
        
        return [dict(zip(['month', 'total_flights', 'delayed_flights', 'avg_arrival_delay', 'delay_percentage'], r)) for r in results]
