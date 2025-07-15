# Healthcare FastAPI Authentication System - Supabase Compatible
# Production-ready with HIPAA-compliant security

import asyncio
import hashlib
import secrets
import re
from datetime import datetime, timedelta, date
from typing import Optional, Dict, Any, List
from enum import Enum
import os
import jwt
from fastapi import FastAPI, HTTPException, Depends, Request, status, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, validator, Field
import uvicorn
from functools import wraps
import time
import logging
from contextlib import asynccontextmanager
from supabase import create_client, Client
import asyncpg
from sqlalchemy import text
import json

# Configure logging for HIPAA compliance
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('healthcare_auth.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration - Use environment variables in production
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "your-supabase-url")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "your-supabase-anon-key")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@db.your-project.supabase.co:5432/postgres")

# Security configurations
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Rate limiting storage (use Redis in production)
rate_limit_storage: Dict[str, List[float]] = {}

# User roles
class UserRole(str, Enum):
    DOCTOR = "doctor"
    NURSE = "nurse"
    ADMIN = "admin"

# Patient enums
class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"

class BloodType(str, Enum):
    A_POSITIVE = "A+"
    A_NEGATIVE = "A-"
    B_POSITIVE = "B+"
    B_NEGATIVE = "B-"
    AB_POSITIVE = "AB+"
    AB_NEGATIVE = "AB-"
    O_POSITIVE = "O+"
    O_NEGATIVE = "O-"

class InsuranceType(str, Enum):
    PRIVATE = "private"
    MEDICARE = "medicare"
    MEDICAID = "medicaid"
    UNINSURED = "uninsured"

class MaritalStatus(str, Enum):
    SINGLE = "single"
    MARRIED = "married"
    DIVORCED = "divorced"
    WIDOWED = "widowed"

# Database models
class User(BaseModel):
    id: int
    email: EmailStr
    full_name: str
    role: UserRole
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    account_locked_until: Optional[datetime] = None

class Patient(BaseModel):
    id: int
    patient_id: str  # Unique patient identifier
    first_name: str
    last_name: str
    date_of_birth: date
    gender: Gender
    phone: Optional[str] = None
    email: Optional[EmailStr] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    emergency_contact_name: Optional[str] = None
    emergency_contact_phone: Optional[str] = None
    insurance_type: Optional[InsuranceType] = None
    insurance_number: Optional[str] = None
    blood_type: Optional[BloodType] = None
    allergies: Optional[str] = None
    medical_history: Optional[str] = None
    current_medications: Optional[str] = None
    marital_status: Optional[MaritalStatus] = None
    occupation: Optional[str] = None
    is_active: bool = True
    created_at: datetime
    updated_at: datetime
    created_by: int  # User ID who created the record
    updated_by: int  # User ID who last updated the record

# Request/Response models (keeping the same from your code)
class UserRegister(BaseModel):
    email: EmailStr
    password: str
    full_name: str = Field(..., min_length=2, max_length=100)
    role: UserRole

    @validator('password')
    def validate_password(cls, v):
        if len(v) < 12:
            raise ValueError('Password must be at least 12 characters long')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must contain at least one special character')
        return v

    @validator('email')
    def validate_email(cls, v):
        if len(v) > 254:
            raise ValueError('Email address too long')
        return v.lower()

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int

class TokenRefresh(BaseModel):
    refresh_token: str

class UserProfile(BaseModel):
    id: int
    email: EmailStr
    full_name: str
    role: UserRole
    is_active: bool
    is_verified: bool
    last_login: Optional[datetime]

class MessageResponse(BaseModel):
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Patient request/response models
class PatientCreate(BaseModel):
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    date_of_birth: date
    gender: Gender
    phone: Optional[str] = Field(None, regex=r'^\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$')
    email: Optional[EmailStr] = None
    address: Optional[str] = Field(None, max_length=200)
    city: Optional[str] = Field(None, max_length=100)
    state: Optional[str] = Field(None, max_length=50)
    zip_code: Optional[str] = Field(None, regex=r'^\d{5}(-\d{4})?$')
    emergency_contact_name: Optional[str] = Field(None, max_length=100)
    emergency_contact_phone: Optional[str] = Field(None, regex=r'^\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$')
    insurance_type: Optional[InsuranceType] = None
    insurance_number: Optional[str] = Field(None, max_length=50)
    blood_type: Optional[BloodType] = None
    allergies: Optional[str] = Field(None, max_length=1000)
    medical_history: Optional[str] = Field(None, max_length=5000)
    current_medications: Optional[str] = Field(None, max_length=2000)
    marital_status: Optional[MaritalStatus] = None
    occupation: Optional[str] = Field(None, max_length=100)

    @validator('date_of_birth')
    def validate_date_of_birth(cls, v):
        if v > date.today():
            raise ValueError('Date of birth cannot be in the future')
        if v < date(1900, 1, 1):
            raise ValueError('Date of birth cannot be before 1900')
        return v

class PatientUpdate(BaseModel):
    first_name: Optional[str] = Field(None, min_length=1, max_length=100)
    last_name: Optional[str] = Field(None, min_length=1, max_length=100)
    phone: Optional[str] = Field(None, regex=r'^\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$')
    email: Optional[EmailStr] = None
    address: Optional[str] = Field(None, max_length=200)
    city: Optional[str] = Field(None, max_length=100)
    state: Optional[str] = Field(None, max_length=50)
    zip_code: Optional[str] = Field(None, regex=r'^\d{5}(-\d{4})?$')
    emergency_contact_name: Optional[str] = Field(None, max_length=100)
    emergency_contact_phone: Optional[str] = Field(None, regex=r'^\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$')
    insurance_type: Optional[InsuranceType] = None
    insurance_number: Optional[str] = Field(None, max_length=50)
    blood_type: Optional[BloodType] = None
    allergies: Optional[str] = Field(None, max_length=1000)
    medical_history: Optional[str] = Field(None, max_length=5000)
    current_medications: Optional[str] = Field(None, max_length=2000)
    marital_status: Optional[MaritalStatus] = None
    occupation: Optional[str] = Field(None, max_length=100)

class PatientResponse(BaseModel):
    id: int
    patient_id: str
    first_name: str
    last_name: str
    date_of_birth: date
    gender: Gender
    phone: Optional[str]
    email: Optional[EmailStr]
    address: Optional[str]
    city: Optional[str]
    state: Optional[str]
    zip_code: Optional[str]
    emergency_contact_name: Optional[str]
    emergency_contact_phone: Optional[str]
    insurance_type: Optional[InsuranceType]
    insurance_number: Optional[str]
    blood_type: Optional[BloodType]
    allergies: Optional[str]
    medical_history: Optional[str]
    current_medications: Optional[str]
    marital_status: Optional[MaritalStatus]
    occupation: Optional[str]
    age: int
    created_at: datetime
    updated_at: datetime

class PaginatedPatients(BaseModel):
    patients: List[PatientResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_previous: bool

class PatientSearchFilters(BaseModel):
    search: Optional[str] = None
    gender: Optional[Gender] = None
    min_age: Optional[int] = Field(None, ge=0, le=150)
    max_age: Optional[int] = Field(None, ge=0, le=150)
    insurance_type: Optional[InsuranceType] = None
    blood_type: Optional[BloodType] = None
    city: Optional[str] = None
    state: Optional[str] = None
    marital_status: Optional[MaritalStatus] = None

# Supabase Database Manager
class SupabaseManager:
    def __init__(self):
        self.client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.pool = None
    
    async def connect(self):
        """Initialize asyncpg connection pool for direct PostgreSQL operations"""
        self.pool = await asyncpg.create_pool(DATABASE_URL)
        await self.create_tables()
    
    async def disconnect(self):
        if self.pool:
            await self.pool.close()
    
    async def create_tables(self):
        """Create tables using direct PostgreSQL connection"""
        async with self.pool.acquire() as conn:
            # Check if tables exist first
            users_table_exists = await conn.fetchval(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'users')"
            )
            
            if not users_table_exists:
                await conn.execute('''
                    CREATE TABLE users (
                        id SERIAL PRIMARY KEY,
                        email VARCHAR(255) UNIQUE NOT NULL,
                        full_name VARCHAR(100) NOT NULL,
                        password_hash VARCHAR(255) NOT NULL,
                        role VARCHAR(20) NOT NULL,
                        is_active BOOLEAN DEFAULT TRUE,
                        is_verified BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_login TIMESTAMP,
                        failed_login_attempts INTEGER DEFAULT 0,
                        account_locked_until TIMESTAMP,
                        verification_token VARCHAR(255),
                        password_reset_token VARCHAR(255),
                        password_reset_expires TIMESTAMP
                    )
                ''')
                
                await conn.execute('''
                    CREATE TABLE refresh_tokens (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER REFERENCES users(id),
                        token_hash VARCHAR(255) NOT NULL,
                        expires_at TIMESTAMP NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        revoked BOOLEAN DEFAULT FALSE
                    )
                ''')
                
                await conn.execute('''
                    CREATE TABLE audit_logs (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER REFERENCES users(id),
                        action VARCHAR(100) NOT NULL,
                        ip_address INET,
                        user_agent TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        details JSONB
                    )
                ''')
                
                # Create indexes for performance
                await conn.execute('CREATE INDEX idx_users_email ON users(email)')
                await conn.execute('CREATE INDEX idx_refresh_tokens_user_id ON refresh_tokens(user_id)')
                await conn.execute('CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id)')
                await conn.execute('CREATE INDEX idx_audit_logs_timestamp ON audit_logs(timestamp)')

            # Create patients table
            patients_table_exists = await conn.fetchval(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'patients')"
            )
            
            if not patients_table_exists:
                await conn.execute('''
                    CREATE TABLE patients (
                        id SERIAL PRIMARY KEY,
                        patient_id VARCHAR(50) UNIQUE NOT NULL,
                        first_name VARCHAR(100) NOT NULL,
                        last_name VARCHAR(100) NOT NULL,
                        date_of_birth DATE NOT NULL,
                        gender VARCHAR(10) NOT NULL,
                        phone VARCHAR(20),
                        email VARCHAR(255),
                        address VARCHAR(200),
                        city VARCHAR(100),
                        state VARCHAR(50),
                        zip_code VARCHAR(10),
                        emergency_contact_name VARCHAR(100),
                        emergency_contact_phone VARCHAR(20),
                        insurance_type VARCHAR(20),
                        insurance_number VARCHAR(50),
                        blood_type VARCHAR(5),
                        allergies TEXT,
                        medical_history TEXT,
                        current_medications TEXT,
                        marital_status VARCHAR(20),
                        occupation VARCHAR(100),
                        is_active BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        created_by INTEGER REFERENCES users(id),
                        updated_by INTEGER REFERENCES users(id)
                    )
                ''')

                # Create indexes for performance and search
                await conn.execute('CREATE INDEX idx_patients_patient_id ON patients(patient_id)')
                await conn.execute('CREATE INDEX idx_patients_last_name ON patients(last_name)')
                await conn.execute('CREATE INDEX idx_patients_first_name ON patients(first_name)')
                await conn.execute('CREATE INDEX idx_patients_date_of_birth ON patients(date_of_birth)')
                await conn.execute('CREATE INDEX idx_patients_gender ON patients(gender)')
                await conn.execute('CREATE INDEX idx_patients_insurance_type ON patients(insurance_type)')
                await conn.execute('CREATE INDEX idx_patients_blood_type ON patients(blood_type)')
                await conn.execute('CREATE INDEX idx_patients_city ON patients(city)')
                await conn.execute('CREATE INDEX idx_patients_state ON patients(state)')
                await conn.execute('CREATE INDEX idx_patients_zip_code ON patients(zip_code)')
                await conn.execute('CREATE INDEX idx_patients_is_active ON patients(is_active)')
                await conn.execute('CREATE INDEX idx_patients_created_at ON patients(created_at)')
                await conn.execute('CREATE INDEX idx_patients_updated_at ON patients(updated_at)')
                
                # Full-text search indexes
                await conn.execute('''
                    CREATE INDEX idx_patients_full_text_search ON patients 
                    USING GIN(to_tsvector('english', 
                        COALESCE(first_name, '') || ' ' || 
                        COALESCE(last_name, '') || ' ' || 
                        COALESCE(patient_id, '') || ' ' ||
                        COALESCE(email, '') || ' ' ||
                        COALESCE(phone, '') || ' ' ||
                        COALESCE(address, '') || ' ' ||
                        COALESCE(city, '') || ' ' ||
                        COALESCE(state, '') || ' ' ||
                        COALESCE(occupation, '')
                    ))
                ''')

                # Create trigger for updating updated_at timestamp
                await conn.execute('''
                    CREATE OR REPLACE FUNCTION update_modified_column()
                    RETURNS TRIGGER AS $$
                    BEGIN
                        NEW.updated_at = CURRENT_TIMESTAMP;
                        RETURN NEW;
                    END;
                    $$ language 'plpgsql';
                ''')

                await conn.execute('''
                    CREATE TRIGGER update_patient_modtime 
                    BEFORE UPDATE ON patients 
                    FOR EACH ROW 
                    EXECUTE FUNCTION update_modified_column();
                ''')

# Initialize Supabase manager
db_manager = SupabaseManager()

# Rate limiting decorator (same as before)
def rate_limit(requests_per_minute: int = 100):
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            client_ip = request.client.host
            current_time = time.time()
            
            if client_ip in rate_limit_storage:
                rate_limit_storage[client_ip] = [
                    timestamp for timestamp in rate_limit_storage[client_ip]
                    if current_time - timestamp < 60
                ]
            else:
                rate_limit_storage[client_ip] = []
            
            if len(rate_limit_storage[client_ip]) >= requests_per_minute:
                logger.warning(f"Rate limit exceeded for IP: {client_ip}")
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded. Please try again later."
                )
            
            rate_limit_storage[client_ip].append(current_time)
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator

# Authentication utilities (same as before)
class AuthManager:
    @staticmethod
    def hash_password(password: str) -> str:
        return pwd_context.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def create_access_token(data: dict) -> str:
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire, "type": "access"})
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    @staticmethod
    def create_refresh_token(data: dict) -> str:
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": "refresh"})
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    @staticmethod
    def verify_token(token: str, token_type: str = "access") -> dict:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            if payload.get("type") != token_type:
                raise HTTPException(status_code=401, detail="Invalid token type")
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")

# User management (same database operations)
class UserManager:
    @staticmethod
    async def get_user_by_email(email: str) -> Optional[User]:
        async with db_manager.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM users WHERE email = $1", email
            )
            if row:
                return User(**dict(row))
            return None
    
    @staticmethod
    async def get_user_by_id(user_id: int) -> Optional[User]:
        async with db_manager.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM users WHERE id = $1", user_id
            )
            if row:
                return User(**dict(row))
            return None
    
    @staticmethod
    async def create_user(user_data: UserRegister) -> User:
        password_hash = AuthManager.hash_password(user_data.password)
        verification_token = secrets.token_urlsafe(32)
        
        async with db_manager.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO users (email, full_name, password_hash, role, verification_token)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING *
                """,
                user_data.email,
                user_data.full_name,
                password_hash,
                user_data.role.value,
                verification_token
            )
            return User(**dict(row))
    
    @staticmethod
    async def update_last_login(user_id: int):
        async with db_manager.pool.acquire() as conn:
            await conn.execute(
                "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = $1",
                user_id
            )
    
    @staticmethod
    async def increment_failed_login(user_id: int):
        async with db_manager.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE users 
                SET failed_login_attempts = failed_login_attempts + 1,
                    account_locked_until = CASE 
                        WHEN failed_login_attempts >= 4 
                        THEN CURRENT_TIMESTAMP + INTERVAL '30 minutes'
                        ELSE account_locked_until
                    END
                WHERE id = $1
                """,
                user_id
            )
    
    @staticmethod
    async def reset_failed_login(user_id: int):
        async with db_manager.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE users 
                SET failed_login_attempts = 0, account_locked_until = NULL
                WHERE id = $1
                """,
                user_id
            )

# Patient Management
class PatientManager:
    @staticmethod
    def _generate_patient_id() -> str:
        """Generate unique patient ID"""
        timestamp = datetime.now().strftime('%Y%m%d')
        random_part = secrets.token_hex(3).upper()
        return f"PAT{timestamp}{random_part}"
    
    @staticmethod
    def _calculate_age(date_of_birth: date) -> int:
        """Calculate age from date of birth"""
        today = date.today()
        return today.year - date_of_birth.year - ((today.month, today.day) < (date_of_birth.month, date_of_birth.day))
    
    @staticmethod
    async def create_patient(patient_data: PatientCreate, user_id: int) -> Patient:
        """Create a new patient record"""
        patient_id = PatientManager._generate_patient_id()
        
        async with db_manager.pool.acquire() as conn:
            # Check if patient_id already exists (very unlikely but let's be safe)
            while await conn.fetchval("SELECT id FROM patients WHERE patient_id = $1", patient_id):
                patient_id = PatientManager._generate_patient_id()
            
            row = await conn.fetchrow(
                """
                INSERT INTO patients (
                    patient_id, first_name, last_name, date_of_birth, gender, phone, email,
                    address, city, state, zip_code, emergency_contact_name, emergency_contact_phone,
                    insurance_type, insurance_number, blood_type, allergies, medical_history,
                    current_medications, marital_status, occupation, created_by, updated_by
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $22
                ) RETURNING *
                """,
                patient_id, patient_data.first_name, patient_data.last_name, patient_data.date_of_birth,
                patient_data.gender.value, patient_data.phone, patient_data.email, patient_data.address,
                patient_data.city, patient_data.state, patient_data.zip_code, patient_data.emergency_contact_name,
                patient_data.emergency_contact_phone, patient_data.insurance_type.value if patient_data.insurance_type else None,
                patient_data.insurance_number, patient_data.blood_type.value if patient_data.blood_type else None,
                patient_data.allergies, patient_data.medical_history, patient_data.current_medications,
                patient_data.marital_status.value if patient_data.marital_status else None, patient_data.occupation,
                user_id
            )
            return Patient(**dict(row))
    
    @staticmethod
    async def get_patient_by_id(patient_id: int) -> Optional[Patient]:
        """Get patient by ID"""
        async with db_manager.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM patients WHERE id = $1 AND is_active = TRUE",
                patient_id
            )
            if row:
                return Patient(**dict(row))
            return None
    
    @staticmethod
    async def get_patient_by_patient_id(patient_id: str) -> Optional[Patient]:
        """Get patient by patient_id"""
        async with db_manager.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM patients WHERE patient_id = $1 AND is_active = TRUE",
                patient_id
            )
            if row:
                return Patient(**dict(row))
            return None
    
    @staticmethod
    async def update_patient(patient_id: int, patient_data: PatientUpdate, user_id: int) -> Optional[Patient]:
        """Update patient record"""
        async with db_manager.pool.acquire() as conn:
            # Build dynamic update query
            update_fields = []
            params = []
            param_count = 1
            
            for field, value in patient_data.dict(exclude_unset=True).items():
                if value is not None:
                    if isinstance(value, Enum):
                        value = value.value
                    update_fields.append(f"{field} = ${param_count}")
                    params.append(value)
                    param_count += 1
            
            if not update_fields:
                # No fields to update
                return await PatientManager.get_patient_by_id(patient_id)
            
            # Add updated_by field
            update_fields.append(f"updated_by = ${param_count}")
            params.append(user_id)
            param_count += 1
            
            # Add patient_id to params
            params.append(patient_id)
            
            query = f"""
                UPDATE patients 
                SET {', '.join(update_fields)}
                WHERE id = ${param_count} AND is_active = TRUE
                RETURNING *
            """
            
            row = await conn.fetchrow(query, *params)
            if row:
                return Patient(**dict(row))
            return None
    
    @staticmethod
    async def soft_delete_patient(patient_id: int, user_id: int) -> bool:
        """Soft delete patient record"""
        async with db_manager.pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE patients 
                SET is_active = FALSE, updated_by = $1
                WHERE id = $2 AND is_active = TRUE
                """,
                user_id, patient_id
            )
            return result == "UPDATE 1"
    
    @staticmethod
    async def search_patients(
        filters: PatientSearchFilters,
        page: int = 1,
        page_size: int = 50
    ) -> tuple[List[Patient], int]:
        """Search patients with filters and pagination"""
        async with db_manager.pool.acquire() as conn:
            # Build WHERE clause
            where_conditions = ["is_active = TRUE"]
            params = []
            param_count = 1
            
            if filters.search:
                where_conditions.append(f"""
                    to_tsvector('english', 
                        COALESCE(first_name, '') || ' ' || 
                        COALESCE(last_name, '') || ' ' || 
                        COALESCE(patient_id, '') || ' ' ||
                        COALESCE(email, '') || ' ' ||
                        COALESCE(phone, '') || ' ' ||
                        COALESCE(address, '') || ' ' ||
                        COALESCE(city, '') || ' ' ||
                        COALESCE(state, '') || ' ' ||
                        COALESCE(occupation, '')
                    ) @@ plainto_tsquery('english', ${param_count})
                """)
                params.append(filters.search)
                param_count += 1
            
            if filters.gender:
                where_conditions.append(f"gender = ${param_count}")
                params.append(filters.gender.value)
                param_count += 1
            
            if filters.insurance_type:
                where_conditions.append(f"insurance_type = ${param_count}")
                params.append(filters.insurance_type.value)
                param_count += 1
            
            if filters.blood_type:
                where_conditions.append(f"blood_type = ${param_count}")
                params.append(filters.blood_type.value)
                param_count += 1
            
            if filters.city:
                where_conditions.append(f"LOWER(city) LIKE LOWER(${param_count})")
                params.append(f"%{filters.city}%")
                param_count += 1
            
            if filters.state:
                where_conditions.append(f"LOWER(state) LIKE LOWER(${param_count})")
                params.append(f"%{filters.state}%")
                param_count += 1
            
            if filters.marital_status:
                where_conditions.append(f"marital_status = ${param_count}")
                params.append(filters.marital_status.value)
                param_count += 1
            
            # Age filtering
            if filters.min_age is not None:
                where_conditions.append(f"date_of_birth <= ${param_count}")
                min_birth_date = date.today() - timedelta(days=filters.min_age * 365.25)
                params.append(min_birth_date)
                param_count += 1
            
            if filters.max_age is not None:
                where_conditions.append(f"date_of_birth >= ${param_count}")
                max_birth_date = date.today() - timedelta(days=(filters.max_age + 1) * 365.25)
                params.append(max_birth_date)
                param_count += 1
            
            where_clause = " AND ".join(where_conditions)
            
            # Count total records
            count_query = f"SELECT COUNT(*) FROM patients WHERE {where_clause}"
            total = await conn.fetchval(count_query, *params)
            
            # Get paginated results
            offset = (page - 1) * page_size
            params.extend([page_size, offset])
            
            search_query = f"""
                SELECT * FROM patients 
                WHERE {where_clause}
                ORDER BY updated_at DESC, last_name ASC, first_name ASC
                LIMIT ${param_count} OFFSET ${param_count + 1}
            """
            
            rows = await conn.fetch(search_query, *params)
            patients = [Patient(**dict(row)) for row in rows]
            
            return patients, total

# Audit Logger
class AuditLogger:
    @staticmethod
    async def log_patient_action(
        user_id: int,
        action: str,
        patient_id: Optional[int] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log patient-related actions for HIPAA compliance"""
        async with db_manager.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO audit_logs (user_id, action, ip_address, user_agent, details)
                VALUES ($1, $2, $3, $4, $5)
                """,
                user_id, action, ip_address, user_agent, json.dumps(details or {})
            )

# Refresh token manager (same as before)
class RefreshTokenManager:
    @staticmethod
    async def store_refresh_token(user_id: int, refresh_token: str):
        token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
        expires_at = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        
        async with db_manager.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO refresh_tokens (user_id, token_hash, expires_at)
                VALUES ($1, $2, $3)
                """,
                user_id, token_hash, expires_at
            )
    
    @staticmethod
    async def verify_refresh_token(refresh_token: str) -> Optional[int]:
        token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
        
        async with db_manager.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT user_id FROM refresh_tokens 
                WHERE token_hash = $1 AND expires_at > CURRENT_TIMESTAMP AND revoked = FALSE
                """,
                token_hash
            )
            return row['user_id'] if row else None
    
    @staticmethod
    async def revoke_refresh_token(refresh_token: str):
        token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
        
        async with db_manager.pool.acquire() as conn:
            await conn.execute(
                "UPDATE refresh_tokens SET revoked = TRUE WHERE token_hash = $1",
                token_hash
            )

# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current authenticated user"""
    try:
        payload = AuthManager.verify_token(credentials.credentials)
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user = await UserManager.get_user_by_id(int(user_id))
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        
        if not user.is_active:
            raise HTTPException(status_code=401, detail="User account is disabled")
        
        return user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")

# Role-based access control
def require_role(allowed_roles: List[UserRole]):
    def decorator(func):
        @wraps(func)
        async def wrapper(current_user: User = Depends(get_current_user), *args, **kwargs):
            if current_user.role not in allowed_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions"
                )
            return await func(current_user, *args, **kwargs)
        return wrapper
    return decorator

# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await db_manager.connect()
    logger.info("Healthcare authentication system started with Supabase")
    yield
    # Shutdown
    await db_manager.disconnect()
    logger.info("Healthcare authentication system stopped")

# FastAPI app
app = FastAPI(
    title="Healthcare Authentication API",
    description="HIPAA-compliant authentication system for healthcare applications",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware with your domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

# Authentication endpoints (keeping existing structure)
@app.post("/auth/register", response_model=UserProfile)
@rate_limit(10)
async def register_user(user_data: UserRegister, request: Request):
    """Register a new user"""
    try:
        # Check if user already exists
        existing_user = await UserManager.get_user_by_email(user_data.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists"
            )
        
        # Create new user
        user = await UserManager.create_user(user_data)
        
        # Log the registration
        await AuditLogger.log_patient_action(
            user_id=user.id,
            action="user_registered",
            ip_address=request.client.host,
            user_agent=request.headers.get("User-Agent"),
            details={"email": user.email, "role": user.role}
        )
        
        return UserProfile(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            is_active=user.is_active,
            is_verified=user.is_verified,
            last_login=user.last_login
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@app.post("/auth/login", response_model=Token)
@rate_limit(20)
async def login(user_data: UserLogin, request: Request):
    """Authenticate user and return tokens"""
    try:
        # Get user by email
        user = await UserManager.get_user_by_email(user_data.email)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        # Check if account is locked
        if user.account_locked_until and user.account_locked_until > datetime.utcnow():
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail="Account temporarily locked due to multiple failed attempts"
            )
        
        # Verify password
        if not AuthManager.verify_password(user_data.password, user.password_hash):
            await UserManager.increment_failed_login(user.id)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        # Reset failed login attempts
        await UserManager.reset_failed_login(user.id)
        await UserManager.update_last_login(user.id)
        
        # Create tokens
        access_token = AuthManager.create_access_token({"sub": str(user.id)})
        refresh_token = AuthManager.create_refresh_token({"sub": str(user.id)})
        
        # Store refresh token
        await RefreshTokenManager.store_refresh_token(user.id, refresh_token)
        
        # Log successful login
        await AuditLogger.log_patient_action(
            user_id=user.id,
            action="user_login",
            ip_address=request.client.host,
            user_agent=request.headers.get("User-Agent")
        )
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@app.get("/auth/me", response_model=UserProfile)
async def get_current_user_profile(current_user: User = Depends(get_current_user)):
    """Get current user profile"""
    return UserProfile(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role,
        is_active=current_user.is_active,
        is_verified=current_user.is_verified,
        last_login=current_user.last_login
    )

# Patient Management Endpoints
@app.post("/patients/", response_model=PatientResponse)
@rate_limit(50)
async def create_patient(
    patient_data: PatientCreate,
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Create a new patient record"""
    try:
        # Check permissions - only doctors, nurses, and admins can create patients
        if current_user.role not in [UserRole.DOCTOR, UserRole.NURSE, UserRole.ADMIN]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to create patient records"
            )
        
        # Create patient
        patient = await PatientManager.create_patient(patient_data, current_user.id)
        
        # Calculate age
        age = PatientManager._calculate_age(patient.date_of_birth)
        
        # Log the action
        await AuditLogger.log_patient_action(
            user_id=current_user.id,
            action="patient_created",
            patient_id=patient.id,
            ip_address=request.client.host,
            user_agent=request.headers.get("User-Agent"),
            details={"patient_id": patient.patient_id, "full_name": f"{patient.first_name} {patient.last_name}"}
        )
        
        return PatientResponse(
            **patient.dict(),
            age=age
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Patient creation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create patient record"
        )

@app.get("/patients/", response_model=PaginatedPatients)
@rate_limit(100)
async def get_patients(
    request: Request,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Number of patients per page"),
    search: Optional[str] = Query(None, description="Search term"),
    gender: Optional[Gender] = Query(None, description="Filter by gender"),
    min_age: Optional[int] = Query(None, ge=0, le=150, description="Minimum age"),
    max_age: Optional[int] = Query(None, ge=0, le=150, description="Maximum age"),
    insurance_type: Optional[InsuranceType] = Query(None, description="Filter by insurance type"),
    blood_type: Optional[BloodType] = Query(None, description="Filter by blood type"),
    city: Optional[str] = Query(None, description="Filter by city"),
    state: Optional[str] = Query(None, description="Filter by state"),
    marital_status: Optional[MaritalStatus] = Query(None, description="Filter by marital status"),
    current_user: User = Depends(get_current_user)
):
    """Get paginated list of patients with search and filtering"""
    try:
        # Create filters object
        filters = PatientSearchFilters(
            search=search,
            gender=gender,
            min_age=min_age,
            max_age=max_age,
            insurance_type=insurance_type,
            blood_type=blood_type,
            city=city,
            state=state,
            marital_status=marital_status
        )
        
        # Search patients
        patients, total = await PatientManager.search_patients(filters, page, page_size)
        
        # Calculate additional fields
        patient_responses = []
        for patient in patients:
            age = PatientManager._calculate_age(patient.date_of_birth)
            patient_responses.append(PatientResponse(**patient.dict(), age=age))
        
        # Calculate pagination info
        total_pages = (total + page_size - 1) // page_size
        has_next = page < total_pages
        has_previous = page > 1
        
        # Log the action
        await AuditLogger.log_patient_action(
            user_id=current_user.id,
            action="patients_list_accessed",
            ip_address=request.client.host,
            user_agent=request.headers.get("User-Agent"),
            details={"page": page, "page_size": page_size, "total_found": total}
        )
        
        return PaginatedPatients(
            patients=patient_responses,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=has_next,
            has_previous=has_previous
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Patient list error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve patient list"
        )

@app.get("/patients/{patient_id}", response_model=PatientResponse)
@rate_limit(100)
async def get_patient(
    patient_id: int,
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Get patient details by ID"""
    try:
        # Get patient
        patient = await PatientManager.get_patient_by_id(patient_id)
        if not patient:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Patient not found"
            )
        
        # Calculate age
        age = PatientManager._calculate_age(patient.date_of_birth)
        
        # Log the action
        await AuditLogger.log_patient_action(
            user_id=current_user.id,
            action="patient_accessed",
            patient_id=patient.id,
            ip_address=request.client.host,
            user_agent=request.headers.get("User-Agent"),
            details={"patient_id": patient.patient_id, "full_name": f"{patient.first_name} {patient.last_name}"}
        )
        
        return PatientResponse(
            **patient.dict(),
            age=age
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Patient retrieval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve patient"
        )

@app.put("/patients/{patient_id}", response_model=PatientResponse)
@rate_limit(50)
async def update_patient(
    patient_id: int,
    patient_data: PatientUpdate,
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Update patient information"""
    try:
        # Check permissions - only doctors, nurses, and admins can update patients
        if current_user.role not in [UserRole.DOCTOR, UserRole.NURSE, UserRole.ADMIN]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to update patient records"
            )
        
        # Update patient
        patient = await PatientManager.update_patient(patient_id, patient_data, current_user.id)
        if not patient:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Patient not found"
            )
        
        # Calculate age
        age = PatientManager._calculate_age(patient.date_of_birth)
        
        # Log the action
        await AuditLogger.log_patient_action(
            user_id=current_user.id,
            action="patient_updated",
            patient_id=patient.id,
            ip_address=request.client.host,
            user_agent=request.headers.get("User-Agent"),
            details={
                "patient_id": patient.patient_id,
                "full_name": f"{patient.first_name} {patient.last_name}",
                "updated_fields": list(patient_data.dict(exclude_unset=True).keys())
            }
        )
        
        return PatientResponse(
            **patient.dict(),
            age=age
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Patient update error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update patient record"
        )

@app.delete("/patients/{patient_id}", response_model=MessageResponse)
@rate_limit(20)
async def delete_patient(
    patient_id: int,
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Soft delete patient record"""
    try:
        # Check permissions - only doctors and admins can delete patients
        if current_user.role not in [UserRole.DOCTOR, UserRole.ADMIN]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to delete patient records"
            )
        
        # Get patient info before deletion for logging
        patient = await PatientManager.get_patient_by_id(patient_id)
        if not patient:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Patient not found"
            )
        
        # Soft delete patient
        success = await PatientManager.soft_delete_patient(patient_id, current_user.id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to delete patient record"
            )
        
        # Log the action
        await AuditLogger.log_patient_action(
            user_id=current_user.id,
            action="patient_deleted",
            patient_id=patient.id,
            ip_address=request.client.host,
            user_agent=request.headers.get("User-Agent"),
            details={
                "patient_id": patient.patient_id,
                "full_name": f"{patient.first_name} {patient.last_name}"
            }
        )
        
        return MessageResponse(message="Patient record successfully deleted")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Patient deletion error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete patient record"
        )

@app.get("/patients/search", response_model=PaginatedPatients)
@rate_limit(100)
async def search_patients(
    request: Request,
    search: str = Query(..., description="Search term (required)"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Number of patients per page"),
    gender: Optional[Gender] = Query(None, description="Filter by gender"),
    min_age: Optional[int] = Query(None, ge=0, le=150, description="Minimum age"),
    max_age: Optional[int] = Query(None, ge=0, le=150, description="Maximum age"),
    insurance_type: Optional[InsuranceType] = Query(None, description="Filter by insurance type"),
    blood_type: Optional[BloodType] = Query(None, description="Filter by blood type"),
    city: Optional[str] = Query(None, description="Filter by city"),
    state: Optional[str] = Query(None, description="Filter by state"),
    marital_status: Optional[MaritalStatus] = Query(None, description="Filter by marital status"),
    current_user: User = Depends(get_current_user)
):
    """Full-text search across patient data"""
    try:
        # Create filters object with search term
        filters = PatientSearchFilters(
            search=search,
            gender=gender,
            min_age=min_age,
            max_age=max_age,
            insurance_type=insurance_type,
            blood_type=blood_type,
            city=city,
            state=state,
            marital_status=marital_status
        )
        
        # Search patients
        patients, total = await PatientManager.search_patients(filters, page, page_size)
        
        # Calculate additional fields
        patient_responses = []
        for patient in patients:
            age = PatientManager._calculate_age(patient.date_of_birth)
            patient_responses.append(PatientResponse(**patient.dict(), age=age))
        
        # Calculate pagination info
        total_pages = (total + page_size - 1) // page_size
        has_next = page < total_pages
        has_previous = page > 1
        
        # Log the action
        await AuditLogger.log_patient_action(
            user_id=current_user.id,
            action="patients_search",
            ip_address=request.client.host,
            user_agent=request.headers.get("User-Agent"),
            details={"search_term": search, "results_found": total}
        )
        
        return PaginatedPatients(
            patients=patient_responses,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=has_next,
            has_previous=has_previous
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Patient search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search patients"
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )