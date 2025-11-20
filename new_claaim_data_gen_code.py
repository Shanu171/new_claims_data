import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# UK PMI Condition Categories with ICD-9 codes
CONDITION_MAPPING = {
    'Musculoskeletal': {
        'codes': ['M54.5', 'M25.5', 'M51.2', 'M17.0', 'M75.1', 'M23.2'],
        'weight': 0.22,
        'avg_claim': 3500,
        'chronic_ratio': 0.15
    },
    'Cardiovascular': {
        'codes': ['I25.1', 'I20.0', 'I48.0', 'I10', 'I50.0'],
        'weight': 0.12,
        'avg_claim': 8500,
        'chronic_ratio': 0.35
    },
    'Digestive': {
        'codes': ['K80.2', 'K35.8', 'K21.0', 'K57.3', 'K40.9'],
        'weight': 0.10,
        'avg_claim': 4200,
        'chronic_ratio': 0.08
    },
    'Oncology': {
        'codes': ['C50.9', 'C61', 'C34.9', 'C18.9', 'C67.9'],
        'weight': 0.08,
        'avg_claim': 15000,
        'chronic_ratio': 0.95
    },
    'Respiratory': {
        'codes': ['J44.0', 'J18.9', 'J45.0', 'J20.9', 'J32.9'],
        'weight': 0.08,
        'avg_claim': 3800,
        'chronic_ratio': 0.25
    },
    'Ophthalmology': {
        'codes': ['H25.9', 'H40.9', 'H35.3', 'H33.0'],
        'weight': 0.07,
        'avg_claim': 2500,
        'chronic_ratio': 0.05
    },
    'Gynaecology': {
        'codes': ['N80.0', 'N92.0', 'N81.1', 'N83.2'],
        'weight': 0.06,
        'avg_claim': 3200,
        'chronic_ratio': 0.10
    },
    'Urology': {
        'codes': ['N20.0', 'N40', 'N39.0', 'N13.3'],
        'weight': 0.05,
        'avg_claim': 4500,
        'chronic_ratio': 0.12
    },
    'Neurology': {
        'codes': ['G43.9', 'G47.0', 'G35', 'G20'],
        'weight': 0.05,
        'avg_claim': 5500,
        'chronic_ratio': 0.40
    },
    'Mental Health': {
        'codes': ['F32.9', 'F41.1', 'F33.2', 'F43.1'],
        'weight': 0.04,
        'avg_claim': 2800,
        'chronic_ratio': 0.30
    },
    'Dermatology': {
        'codes': ['L70.0', 'L40.9', 'L02.9', 'L30.9'],
        'weight': 0.03,
        'avg_claim': 1500,
        'chronic_ratio': 0.08
    },
    'ENT': {
        'codes': ['J35.0', 'H66.9', 'J34.2', 'H81.0'],
        'weight': 0.03,
        'avg_claim': 2200,
        'chronic_ratio': 0.05
    },
    'Endocrine': {
        'codes': ['E11.9', 'E05.0', 'E66.9', 'E78.5'],
        'weight': 0.02,
        'avg_claim': 3500,
        'chronic_ratio': 0.65
    },
    'Rheumatology': {
        'codes': ['M05.9', 'M32.9', 'M06.9', 'M45'],
        'weight': 0.02,
        'avg_claim': 4800,
        'chronic_ratio': 0.80
    },
    'Orthopaedics': {
        'codes': ['S72.0', 'M84.4', 'S82.0', 'M23.5'],
        'weight': 0.01,
        'avg_claim': 7500,
        'chronic_ratio': 0.10
    },
    'Haematology': {
        'codes': ['D50.9', 'D64.9', 'D68.3'],
        'weight': 0.01,
        'avg_claim': 4500,
        'chronic_ratio': 0.45
    },
    'Gastroenterology': {
        'codes': ['K50.9', 'K51.9', 'K29.7'],
        'weight': 0.005,
        'avg_claim': 5200,
        'chronic_ratio': 0.70
    },
    'Maternity': {
        'codes': ['O80', 'O34.2', 'O60.1'],
        'weight': 0.003,
        'avg_claim': 4500,
        'chronic_ratio': 0.00
    },
    'Other': {
        'codes': ['R50.9', 'R07.4', 'R10.4', 'T14.9'],
        'weight': 0.002,
        'avg_claim': 2000,
        'chronic_ratio': 0.05
    }
}

# UK PMI Ancillary Service Types
ANCILLARY_SERVICES = [
    'Physiotherapy',
    'Mental Health Therapy',
    'Diagnostic Imaging',
    'Pathology Tests',
    'Radiotherapy',
    'Chemotherapy',
    'Home Nursing',
    'Medical Appliances',
    'Optical Services'
]

CLAIM_TYPES = ['Inpatient', 'Outpatient', 'Day Case', 'Cash Benefit']
PROVIDER_TYPES = ['Hospital', 'Clinic', 'Diagnostic Center', 'Rehab Center']
TREATMENT_LOCATIONS = ['England', 'Scotland', 'Wales', 'Northern Ireland']

def generate_claim_id(idx):
    """Generate unique claim ID"""
    return f"CLM{str(idx).zfill(8)}"

def get_impairment_code(condition_code):
    """Generate impairment code based on condition"""
    prefix_map = {
        'M': 'IMP-MSK',
        'I': 'IMP-CVD',
        'K': 'IMP-DIG',
        'C': 'IMP-ONC',
        'J': 'IMP-RSP',
        'H': 'IMP-OPH',
        'N': 'IMP-URO',
        'G': 'IMP-NEU',
        'F': 'IMP-MEN',
        'L': 'IMP-DER',
        'E': 'IMP-END',
        'D': 'IMP-HAE',
        'S': 'IMP-ORT',
        'O': 'IMP-MAT',
        'R': 'IMP-GEN',
        'T': 'IMP-GEN'
    }
    prefix = condition_code[0] if condition_code else 'R'
    return f"{prefix_map.get(prefix, 'IMP-GEN')}-{random.randint(100, 999)}"

def calculate_claim_amount(condition_category, claim_type, treatment_type, age):
    """Calculate realistic claim amount based on type, treatment, and age"""
    base_amount = CONDITION_MAPPING[condition_category]['avg_claim']
    
    # Adjust by claim type
    type_multiplier = {
        'Inpatient': 1.8,
        'Day Case': 0.9,
        'Outpatient': 0.4,
        'Cash Benefit': 0.3
    }
    
    # Add variation
    variation = np.random.normal(1.0, 0.3)
    variation = max(0.3, min(variation, 2.5))  # Limit extreme values
    
    # Apply age multiplier
    age_multiplier = get_age_amount_multiplier(age)
    
    amount = base_amount * type_multiplier.get(claim_type, 1.0) * variation * age_multiplier
    
    # Add surgical premium
    if treatment_type == 'Surgical':
        amount *= random.uniform(1.2, 1.8)
    
    return round(amount, 2)

def generate_dates(contract_start, contract_end):
    """Generate incurred and paid dates within contract period"""
    if pd.isna(contract_start) or pd.isna(contract_end):
        return None, None
    
    try:
        contract_start = pd.to_datetime(contract_start, dayfirst=True)
        contract_end = pd.to_datetime(contract_end, dayfirst=True)
    except:
        return None, None
    
    if contract_end <= contract_start:
        return None, None
    
    # Generate incurred date
    days_range = (contract_end - contract_start).days
    if days_range <= 0:
        return None, None
    
    incurred_date = contract_start + timedelta(days=random.randint(0, days_range))
    
    # Paid date is 5-60 days after incurred (realistic processing time)
    processing_days = random.randint(5, 60)
    paid_date = incurred_date + timedelta(days=processing_days)
    
    # Ensure paid date doesn't exceed contract end
    if paid_date > contract_end:
        paid_date = contract_end
    
    return incurred_date, paid_date

def get_age_multiplier(age):
    """Calculate claim frequency multiplier based on age"""
    if age < 18:
        return 0.3  # Children: fewer claims
    elif age < 30:
        return 0.5  # Young adults: low claims
    elif age < 40:
        return 0.7  # 30s: moderate claims
    elif age < 50:
        return 1.0  # 40s: baseline
    elif age < 60:
        return 1.4  # 50s: increased claims
    elif age < 70:
        return 1.8  # 60s: higher claims
    else:
        return 2.2  # 70+: highest claims

def get_age_amount_multiplier(age):
    """Calculate claim amount multiplier based on age"""
    if age < 18:
        return 0.7  # Children: lower costs
    elif age < 30:
        return 0.8  # Young adults: lower costs
    elif age < 40:
        return 1.0  # 30s: baseline
    elif age < 50:
        return 1.1  # 40s: slightly higher
    elif age < 60:
        return 1.3  # 50s: higher costs
    elif age < 70:
        return 1.5  # 60s: much higher costs
    else:
        return 1.8  # 70+: highest costs

def get_age_based_conditions(age):
    """Adjust condition probabilities based on age"""
    if age < 18:
        # Children: more respiratory, ENT, less cardiovascular/oncology
        return {
            'Respiratory': 0.25,
            'ENT': 0.15,
            'Digestive': 0.12,
            'Musculoskeletal': 0.10,
            'Dermatology': 0.08,
            'Ophthalmology': 0.05,
            'Mental Health': 0.03,
            'Other': 0.22
        }
    elif age < 40:
        # Young adults: musculoskeletal, mental health, gynaecology
        return {
            'Musculoskeletal': 0.25,
            'Mental Health': 0.12,
            'Gynaecology': 0.10,
            'Digestive': 0.10,
            'Respiratory': 0.08,
            'Dermatology': 0.05,
            'ENT': 0.05,
            'Urology': 0.03,
            'Other': 0.22
        }
    elif age < 60:
        # Middle age: standard distribution with some cardiovascular
        return {
            'Musculoskeletal': 0.22,
            'Cardiovascular': 0.10,
            'Digestive': 0.10,
            'Gynaecology': 0.08,
            'Urology': 0.06,
            'Respiratory': 0.08,
            'Mental Health': 0.05,
            'Ophthalmology': 0.08,
            'Endocrine': 0.03,
            'Other': 0.20
        }
    else:
        # Older adults: cardiovascular, oncology, orthopaedics dominant
        return {
            'Cardiovascular': 0.20,
            'Oncology': 0.15,
            'Musculoskeletal': 0.15,
            'Ophthalmology': 0.10,
            'Orthopaedics': 0.08,
            'Respiratory': 0.08,
            'Urology': 0.07,
            'Endocrine': 0.05,
            'Neurology': 0.05,
            'Rheumatology': 0.04,
            'Other': 0.03
        }
    """Determine if condition is chronic based on probabilities"""
    chronic_ratio = CONDITION_MAPPING[condition_category]['chronic_ratio']
    return random.random() < chronic_ratio

def generate_claims_data(membership_df, claims_per_member_range=(1, 8), start_year=2018, end_year=2024):
    """Generate claims data for membership table across multiple years"""
    
    claims_data = []
    claim_counter = 1
    
    print(f"Total members in file: {len(membership_df)}")
    print(f"Available columns: {membership_df.columns.tolist()}\n")
    
    # Get active members - based on your data, Registration Status contains "Active"
    # Filter for active members with valid dates
    active_members = membership_df[
        (membership_df['Registration Status'].str.strip() == 'Active')
    ].copy()
    
    print(f"Active members found: {len(active_members)}")
    
    if len(active_members) == 0:
        print("ERROR: No active members found!")
        print("Please check 'Registration Status' column values")
        return pd.DataFrame()
    
    # Further filter for members with valid contract dates
    active_members['Contract Start Date'] = pd.to_datetime(active_members['Contract Start Date'], dayfirst=True, errors='coerce')
    active_members['Contract End Date'] = pd.to_datetime(active_members['Contract End Date'], dayfirst=True, errors='coerce')
    
    valid_members = active_members[
        (active_members['Contract Start Date'].notna()) & 
        (active_members['Contract End Date'].notna()) &
        (active_members['Contract End Date'] > active_members['Contract Start Date'])
    ].copy()
    
    print(f"Members with valid contract dates: {len(valid_members)}")
    
    if len(valid_members) == 0:
        print("ERROR: No members with valid contract dates!")
        return pd.DataFrame()
    
    # Determine utilization rate (60-70% of members make claims in UK PMI)
    utilization_rate = 0.65
    claiming_members = valid_members.sample(frac=utilization_rate, random_state=42)
    
    print(f"Members selected for claims (65% utilization): {len(claiming_members)}")
    print(f"Generating claims for years {start_year} to {end_year}...\n")
    
    # Process each member for each year
    total_iterations = len(claiming_members) * (end_year - start_year + 1)
    processed = 0
    
    for idx, member in claiming_members.iterrows():
        unique_member_ref = str(member['Unique Member Reference'])
        unique_id = str(member['Unique ID'])
        gender = member['Gender']
        year_of_birth = member['Year of Birth']
        
        # Generate claims for each year from 2018 to 2024
        for year in range(start_year, end_year + 1):
            processed += 1
            
            # Set contract dates for this specific year
            contract_start = pd.to_datetime(f'01/01/{year}', dayfirst=True)
            contract_end = pd.to_datetime(f'31/12/{year}', dayfirst=True)
            
            # Calculate age for this year
            age = year - year_of_birth if not pd.isna(year_of_birth) else 40
        
        # Get age-based condition probabilities
        age_condition_weights = get_age_based_conditions(age)
        
        # Select condition category based on age-adjusted weights
        categories = list(age_condition_weights.keys())
        weights = list(age_condition_weights.values())
        
        # Fill in missing categories with small weight
        all_categories = list(CONDITION_MAPPING.keys())
        for cat in all_categories:
            if cat not in categories:
                categories.append(cat)
                weights.append(0.001)
        
        condition_category = random.choices(categories, weights=weights)[0]
        
        # Check if chronic condition
        is_chronic = is_chronic_condition(condition_category)
        
        # Calculate age multiplier for claim frequency
        age_multiplier = get_age_multiplier(age)
        
        # Determine number of claims (adjusted by age)
        if is_chronic:
            # Chronic conditions: 12 claims per year (monthly management)
            years_active = max(1, (contract_end - contract_start).days / 365)
            num_claims = int(12 * years_active * age_multiplier)
        else:
            # Acute conditions: 1-8 claims depending on severity and age
            base_claims = random.randint(*claims_per_member_range)
            num_claims = max(1, int(base_claims * age_multiplier))
        
        # Track claims for this member to avoid overlaps
        member_claims = []
        
        # Generate claims for this member
        for claim_num in range(num_claims):
            # Try to find a non-overlapping date (max 10 attempts)
            incurred_date = None
            paid_date = None
            
            for attempt in range(10):
                temp_incurred, temp_paid = generate_dates(contract_start, contract_end)
                
                if temp_incurred is None or temp_paid is None:
                    continue
                
                # Check for overlaps with existing claims
                has_overlap = False
                for existing_claim in member_claims:
                    existing_incurred = existing_claim['incurred']
                    existing_discharge = existing_claim.get('discharge', existing_claim['incurred'])
                    
                    # Check if new claim overlaps with existing claim period
                    if existing_incurred <= temp_incurred <= existing_discharge:
                        has_overlap = True
                        break
                    if existing_incurred <= temp_paid <= existing_discharge:
                        has_overlap = True
                        break
                    # Check reverse overlap
                    if temp_incurred <= existing_incurred <= temp_paid:
                        has_overlap = True
                        break
                
                if not has_overlap:
                    incurred_date = temp_incurred
                    paid_date = temp_paid
                    break
            
            # Skip if couldn't find non-overlapping date
            if incurred_date is None or paid_date is None:
                continue
            
            # Select condition code from category
            condition_code = random.choice(CONDITION_MAPPING[condition_category]['codes'])
            impairment_code = get_impairment_code(condition_code)
            
            # Determine claim type based on condition severity
            if condition_category in ['Oncology', 'Cardiovascular', 'Neurology']:
                claim_type = random.choices(
                    CLAIM_TYPES, 
                    weights=[0.5, 0.2, 0.2, 0.1]
                )[0]
            else:
                claim_type = random.choices(
                    CLAIM_TYPES, 
                    weights=[0.2, 0.4, 0.3, 0.1]
                )[0]
            
            # Treatment type based on condition category
            if condition_category in ['Oncology', 'Cardiovascular', 'Orthopaedics', 'Urology', 'Gynaecology']:
                treatment_type = random.choices(
                    ['Surgical', 'Medical', 'Diagnostic', 'Therapeutic'],
                    weights=[0.5, 0.3, 0.15, 0.05]
                )[0]
            elif condition_category in ['Mental Health']:
                treatment_type = 'Therapeutic'
            elif condition_category in ['Ophthalmology', 'Dermatology', 'ENT']:
                treatment_type = random.choices(
                    ['Surgical', 'Medical', 'Diagnostic', 'Therapeutic'],
                    weights=[0.4, 0.25, 0.25, 0.1]
                )[0]
            else:
                treatment_type = random.choices(
                    ['Surgical', 'Medical', 'Diagnostic', 'Therapeutic'],
                    weights=[0.3, 0.4, 0.2, 0.1]
                )[0]
            
            # Ancillary service based on condition category and treatment type
            ancillary_service = None
            if random.random() < 0.4:  # 40% of claims have ancillary services
                if condition_category == 'Musculoskeletal':
                    ancillary_service = 'Physiotherapy'
                elif condition_category == 'Mental Health':
                    ancillary_service = 'Mental Health Therapy'
                elif condition_category in ['Oncology', 'Haematology']:
                    ancillary_service = random.choice(['Radiotherapy', 'Chemotherapy', 'Home Nursing'])
                elif condition_category == 'Ophthalmology':
                    ancillary_service = 'Optical Services'
                elif treatment_type == 'Diagnostic':
                    ancillary_service = random.choice(['Diagnostic Imaging', 'Pathology Tests'])
                elif treatment_type == 'Therapeutic':
                    ancillary_service = random.choice(['Physiotherapy', 'Home Nursing'])
                else:
                    ancillary_service = random.choice(['Diagnostic Imaging', 'Pathology Tests', 'Medical Appliances'])
            
            # Provider type based on claim type and treatment type
            if claim_type == 'Inpatient' or (treatment_type == 'Surgical' and claim_type != 'Outpatient'):
                provider_type = 'Hospital'
            elif treatment_type == 'Diagnostic':
                provider_type = random.choice(['Diagnostic Center', 'Clinic', 'Hospital'])
            elif condition_category == 'Mental Health':
                provider_type = random.choice(['Clinic', 'Hospital'])
            elif claim_type == 'Day Case':
                provider_type = random.choice(['Hospital', 'Clinic'])
            elif treatment_type == 'Therapeutic':
                provider_type = random.choice(['Rehab Center', 'Clinic'])
            else:
                provider_type = random.choices(
                    PROVIDER_TYPES,
                    weights=[0.4, 0.35, 0.15, 0.1]
                )[0]
            
            # Treatment location
            treatment_location = random.choices(
                TREATMENT_LOCATIONS,
                weights=[0.85, 0.08, 0.05, 0.02]
            )[0]
            
            # Admission and discharge dates (for inpatient/day case)
            admission_date = incurred_date if claim_type in ['Inpatient', 'Day Case'] else None
            
            if claim_type == 'Inpatient':
                los = random.randint(1, 14)  # Length of stay
                discharge_date = admission_date + timedelta(days=los)
                # Ensure discharge doesn't exceed contract end
                if discharge_date > contract_end:
                    discharge_date = contract_end
                    los = (discharge_date - admission_date).days
            elif claim_type == 'Day Case':
                discharge_date = admission_date
                los = 1
            else:
                discharge_date = None
                los = 0
            
            # Store claim info to check for future overlaps
            claim_info = {
                'incurred': incurred_date,
                'paid': paid_date,
                'discharge': discharge_date if discharge_date else incurred_date
            }
            member_claims.append(claim_info)
            
            # Calculate claim amount (with age factor)
            claim_amount = calculate_claim_amount(condition_category, claim_type, treatment_type, age)
            
            # Amount paid (typically 80-100% of claim amount based on policy excess/co-pay)
            payment_ratio = random.uniform(0.8, 1.0)
            amount_paid = round(claim_amount * payment_ratio, 2)
            
            # Create claim record
            claim_record = {
                'Claimant Unique ID': unique_id,
                'Unique Member Reference': unique_member_ref,
                'Claim ID': generate_claim_id(claim_counter),
                'Incurred Date': incurred_date.strftime('%d/%m/%Y'),
                'Paid Date': paid_date.strftime('%d/%m/%Y'),
                'Condition Code': condition_code,
                'Impairment Code': impairment_code,
                'Condition Category': condition_category,
                'Treatment Type': treatment_type,
                'Claim Type': claim_type,
                'Ancillary Service Type': ancillary_service,
                'Treatment Location': treatment_location,
                'Provider Type': provider_type,
                'Admission Date': admission_date.strftime('%d/%m/%Y') if admission_date else None,
                'Discharge Date': discharge_date.strftime('%d/%m/%Y') if discharge_date else None,
                'Calculate Length of Service': los,
                'Claim Amount': claim_amount,
                'Amount Paid': amount_paid
            }
            
            claims_data.append(claim_record)
            claim_counter += 1
        
        if (idx - claiming_members.index[0]) % 5000 == 0 and idx != claiming_members.index[0]:
            print(f"Processed {idx - claiming_members.index[0]} members, generated {len(claims_data)} claims...")
    
    claims_df = pd.DataFrame(claims_data)
    
    if len(claims_df) == 0:
        print("\nERROR: No claims were generated!")
        return pd.DataFrame()
    
    # Sort by dates
    claims_df['Incurred Date Temp'] = pd.to_datetime(claims_df['Incurred Date'], dayfirst=True)
    claims_df = claims_df.sort_values(['Claimant Unique ID', 'Incurred Date Temp'])
    claims_df = claims_df.drop('Incurred Date Temp', axis=1)
    
    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Total claims generated: {len(claims_df):,}")
    print(f"Unique claimants: {claims_df['Claimant Unique ID'].nunique():,}")
    print(f"Unique members: {claims_df['Unique Member Reference'].nunique():,}")
    print(f"Average claims per claimant: {len(claims_df) / claims_df['Claimant Unique ID'].nunique():.2f}")
    print(f"Total claim amount: £{claims_df['Claim Amount'].sum():,.2f}")
    print(f"Total amount paid: £{claims_df['Amount Paid'].sum():,.2f}")
    print(f"\nClaim Type Distribution:")
    print(claims_df['Claim Type'].value_counts())
    print(f"\nTop 5 Condition Categories:")
    print(claims_df['Condition Category'].value_counts().head())
    
    return claims_df

# Usage example:
print("="*60)
print("UK PMI CLAIMS DATA GENERATOR")
print("="*60)
print("\nTo use:")
print("1. membership_df = pd.read_csv('uk_pmi_membership_120k.csv')")
print("2. claims_df = generate_claims_data(membership_df)")
print("3. claims_df.to_csv('claims_data.csv', index=False)")
print("="*60)
