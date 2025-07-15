# Healthcare Patient Management System

A comprehensive FastAPI-based healthcare patient management system with HIPAA-compliant security, built for handling 100,000+ patient records with high performance and scalability.

## 🚀 Features

### Core Features
- **Complete Patient Management**: Full CRUD operations for patient records
- **Advanced Search & Filtering**: Full-text search across patient data with multiple filters
- **High Performance**: Optimized for handling 100,000+ patient records
- **HIPAA Compliance**: Comprehensive audit logging and security measures
- **Role-Based Access Control**: Different permissions for doctors, nurses, and admins
- **Scalable Architecture**: Built with async/await for high concurrency

### Technical Features
- **FastAPI**: Modern, fast web framework for building APIs
- **Supabase Integration**: PostgreSQL database with real-time capabilities
- **JWT Authentication**: Secure token-based authentication
- **Rate Limiting**: Protection against API abuse
- **Comprehensive Logging**: Detailed audit trail for all operations
- **Input Validation**: Robust data validation with Pydantic models
- **Pagination**: Efficient handling of large datasets

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PostgreSQL 13+ (via Supabase)
- Supabase account and project

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd healthcare-patient-management
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Configuration**
   Create a `.env` file in the root directory:
   ```env
   SECRET_KEY=your-secret-key-here
   SUPABASE_URL=your-supabase-url
   SUPABASE_KEY=your-supabase-anon-key
   DATABASE_URL=postgresql://postgres:password@db.your-project.supabase.co:5432/postgres
   ```

4. **Run the application**
   ```bash
   python main.py
   ```

The API will be available at `http://localhost:8000`

## 📖 API Documentation

### Authentication Endpoints

#### Register User
```http
POST /auth/register
Content-Type: application/json

{
  "email": "doctor@hospital.com",
  "password": "SecurePassword123!",
  "full_name": "Dr. Jane Smith",
  "role": "doctor"
}
```

#### Login
```http
POST /auth/login
Content-Type: application/json

{
  "email": "doctor@hospital.com",
  "password": "SecurePassword123!"
}
```

#### Get Current User
```http
GET /auth/me
Authorization: Bearer {access_token}
```

### Patient Management Endpoints

#### Create Patient
```http
POST /patients/
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "first_name": "John",
  "last_name": "Doe",
  "date_of_birth": "1990-01-15",
  "gender": "male",
  "phone": "555-123-4567",
  "email": "john.doe@email.com",
  "address": "123 Main St",
  "city": "New York",
  "state": "NY",
  "zip_code": "10001",
  "emergency_contact_name": "Jane Doe",
  "emergency_contact_phone": "555-987-6543",
  "insurance_type": "private",
  "insurance_number": "INS123456789",
  "blood_type": "A+",
  "allergies": "Penicillin",
  "medical_history": "Hypertension, Diabetes",
  "current_medications": "Metformin, Lisinopril",
  "marital_status": "married",
  "occupation": "Software Engineer"
}
```

#### Get Patients (with pagination and filtering)
```http
GET /patients/?page=1&page_size=50&search=john&gender=male&min_age=25&max_age=65
Authorization: Bearer {access_token}
```

#### Get Patient by ID
```http
GET /patients/{patient_id}
Authorization: Bearer {access_token}
```

#### Update Patient
```http
PUT /patients/{patient_id}
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "phone": "555-999-8888",
  "address": "456 Oak Ave",
  "current_medications": "Metformin, Lisinopril, Aspirin"
}
```

#### Delete Patient (Soft Delete)
```http
DELETE /patients/{patient_id}
Authorization: Bearer {access_token}
```

#### Search Patients
```http
GET /patients/search?search=diabetes&insurance_type=medicare&page=1&page_size=20
Authorization: Bearer {access_token}
```

### Available Filters

- **search**: Full-text search across name, patient ID, email, phone, address, city, state, occupation
- **gender**: male, female, other
- **min_age**: Minimum age (0-150)
- **max_age**: Maximum age (0-150)
- **insurance_type**: private, medicare, medicaid, uninsured
- **blood_type**: A+, A-, B+, B-, AB+, AB-, O+, O-
- **city**: City name (partial match)
- **state**: State name (partial match)
- **marital_status**: single, married, divorced, widowed

## 🔐 Security Features

### Authentication & Authorization
- JWT-based authentication with access and refresh tokens
- Role-based access control (Doctor, Nurse, Admin)
- Account lockout after failed login attempts
- Password strength validation

### HIPAA Compliance
- Comprehensive audit logging for all patient operations
- IP address and user agent tracking
- Secure password hashing with bcrypt
- Rate limiting to prevent abuse

### Data Protection
- Input validation and sanitization
- SQL injection prevention
- XSS protection
- CORS configuration

## 📊 Performance Optimizations

### Database Optimizations
- **Comprehensive Indexing**: Indexes on frequently queried fields
- **Full-text Search**: PostgreSQL GIN indexes for fast text search
- **Connection Pooling**: Efficient database connection management
- **Optimized Queries**: Efficient SQL queries with proper joins

### API Performance
- **Async Operations**: Non-blocking database operations
- **Pagination**: Efficient large dataset handling
- **Response Caching**: Optimized for concurrent users
- **Rate Limiting**: Prevents API abuse and ensures fair usage

## 🔄 Data Models

### Patient Model
```python
class Patient(BaseModel):
    id: int
    patient_id: str  # Auto-generated unique identifier
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
    is_active: bool = True
    created_at: datetime
    updated_at: datetime
    created_by: int
    updated_by: int
```

### User Roles
- **Doctor**: Full access to patient records (create, read, update, delete)
- **Nurse**: Can create, read, and update patient records
- **Admin**: Full system access including user management

## 🚦 Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `201`: Created
- `400`: Bad Request (validation errors)
- `401`: Unauthorized (authentication required)
- `403`: Forbidden (insufficient permissions)
- `404`: Not Found
- `423`: Locked (account locked)
- `429`: Too Many Requests (rate limited)
- `500`: Internal Server Error

## 📝 Logging & Monitoring

### Audit Logs
All patient operations are logged with:
- User ID and role
- Action performed
- Patient ID (when applicable)
- IP address
- User agent
- Timestamp
- Additional details

### Health Check
```http
GET /health
```

Returns system health status and timestamp.

## 🔧 Configuration

### Environment Variables
- `SECRET_KEY`: JWT signing secret
- `SUPABASE_URL`: Supabase project URL
- `SUPABASE_KEY`: Supabase anon key
- `DATABASE_URL`: PostgreSQL connection string

### Rate Limiting
- Authentication endpoints: 10-20 requests/minute
- Patient operations: 50-100 requests/minute
- Search operations: 100 requests/minute

## 📊 Database Schema

### Tables Created
- `users`: User accounts and authentication
- `patients`: Patient records
- `refresh_tokens`: JWT refresh tokens
- `audit_logs`: Comprehensive audit trail

### Indexes
- Primary keys and foreign keys
- Search optimization indexes
- Full-text search indexes
- Performance indexes on frequently queried fields

## 🚀 Production Deployment

### Recommendations
1. **Database**: Use Supabase Pro or dedicated PostgreSQL instance
2. **Caching**: Implement Redis for rate limiting and session management
3. **Load Balancing**: Use multiple API instances behind a load balancer
4. **Monitoring**: Set up application monitoring and alerting
5. **Backup**: Regular database backups and disaster recovery plan

### Security Checklist
- [ ] Use strong SECRET_KEY in production
- [ ] Configure CORS for your domain
- [ ] Set up HTTPS/TLS
- [ ] Implement proper logging and monitoring
- [ ] Regular security updates
- [ ] Database encryption at rest and in transit

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Check the API documentation at `http://localhost:8000/docs`
- Review the comprehensive error messages
- Check the audit logs for troubleshooting
- Contact the development team

---

**Note**: This system is designed for healthcare environments and includes HIPAA-compliant features. Ensure proper configuration and security measures are in place before deploying in production.