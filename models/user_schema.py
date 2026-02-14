from sqlalchemy import Column, String, DateTime, Boolean
from datetime import datetime
from config.database import Base  # ✅ Use shared Base

class User(Base):
    """User model for authentication"""
    __tablename__ = 'users'
    
    id = Column(String(36), primary_key=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    
    def __repr__(self):
        return f'<User {self.email}>'
