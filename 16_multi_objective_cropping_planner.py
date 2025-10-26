"""
Phase 6, Cell 3: Multi-Objective Cropping Planner (MILP/CP-SAT)
This cell implements optimization-based crop planning with multiple objectives and constraints
"""

import pandas as pd
import numpy as np
import json
import os
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("Setting up Multi-Objective Cropping Planner...")

class MultiObjectiveCroppingPlanner:
    """
    Advanced cropping planner using Mixed Integer Linear Programming (MILP)
    and Constraint Programming (CP-SAT) for optimization
    """
    
    def __init__(self, constraint_engine, semantic_retriever):
        self.constraint_engine = constraint_engine
        self.semantic_retriever = semantic_retriever
        
        # Economic parameters for Uganda/East Africa
        self.crop_economics = {
            'maize': {
                'yield_per_hectare': (2.5, 4.0),  # tons/ha (min, max)
                'price_per_ton': 450000,  # UGX per ton
                'production_cost_per_hectare': 1200000,  # UGX
                'market_demand': 'high',
                'storage_life': 12,  # months
                'processing_requirements': 'minimal'
            },
            'rice': {
                'yield_per_hectare': (3.0, 5.5),
                'price_per_ton': 1800000,
                'production_cost_per_hectare': 2000000,
                'market_demand': 'high',
                'storage_life': 18,
                'processing_requirements': 'moderate'
            },
            'beans': {
                'yield_per_hectare': (1.2, 2.0),
                'price_per_ton': 3000000,
                'production_cost_per_hectare': 800000,
                'market_demand': 'high',
                'storage_life': 24,
                'processing_requirements': 'minimal'
            },
            'cassava': {
                'yield_per_hectare': (15.0, 25.0),
                'price_per_ton': 200000,
                'production_cost_per_hectare': 600000,
                'market_demand': 'medium',
                'storage_life': 6,
                'processing_requirements': 'moderate'
            },
            'sweet_potato': {
                'yield_per_hectare': (8.0, 15.0),
                'price_per_ton': 300000,
                'production_cost_per_hectare': 700000,
                'market_demand': 'medium',
                'storage_life': 4,
                'processing_requirements': 'minimal'
            },
            'banana': {
                'yield_per_hectare': (20.0, 35.0),
                'price_per_ton': 400000,
                'production_cost_per_hectare': 1500000,
                'market_demand': 'high',
                'storage_life': 2,
                'processing_requirements': 'minimal'
            },
            'coffee': {
                'yield_per_hectare': (0.8, 1.5),
                'price_per_ton': 15000000,
                'production_cost_per_hectare': 3000000,
                'market_demand': 'high',
                'storage_life': 36,
                'processing_requirements': 'high'
            },
            'cotton': {
                'yield_per_hectare': (1.5, 2.5),
                'price_per_ton': 2500000,
                'production_cost_per_hectare': 1800000,
                'market_demand': 'medium',
                'storage_life': 12,
                'processing_requirements': 'high'
            },
            'sugarcane': {
                'yield_per_hectare': (60.0, 100.0),
                'price_per_ton': 150000,
                'production_cost_per_hectare': 2500000,
                'market_demand': 'high',
                'storage_life': 1,
                'processing_requirements': 'high'
            },
            'groundnut': {
                'yield_per_hectare': (1.0, 2.0),
                'price_per_ton': 4000000,
                'production_cost_per_hectare': 900000,
                'market_demand': 'high',
                'storage_life': 12,
                'processing_requirements': 'moderate'
            }
        }
        
        # Seasonal constraints
        self.seasonal_constraints = {
            'maize': {'planting_months': [3, 4, 5, 8, 9], 'harvest_months': [7, 8, 11, 12]},
            'rice': {'planting_months': [3, 4, 8, 9], 'harvest_months': [7, 8, 11, 12]},
            'beans': {'planting_months': [3, 4, 8, 9], 'harvest_months': [6, 7, 11, 12]},
            'cassava': {'planting_months': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'harvest_months': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]},
            'sweet_potato': {'planting_months': [3, 4, 5, 8, 9], 'harvest_months': [7, 8, 11, 12]},
            'banana': {'planting_months': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'harvest_months': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]},
            'coffee': {'planting_months': [3, 4, 5], 'harvest_months': [10, 11, 12, 1, 2]},
            'cotton': {'planting_months': [4, 5, 6], 'harvest_months': [10, 11, 12]},
            'sugarcane': {'planting_months': [3, 4, 5, 6], 'harvest_months': [10, 11, 12, 1, 2]},
            'groundnut': {'planting_months': [3, 4, 5, 8, 9], 'harvest_months': [6, 7, 11, 12]}
        }
        
        print("✅ Multi-Objective Cropping Planner initialized")
    
    def optimize_crop_allocation(self, soil_properties, climate_conditions, 
                                available_land=1.0, budget_limit=None, 
                                objectives=['profit', 'food_security', 'risk_diversification'],
                                time_horizon_months=12):
        """
        Optimize crop allocation using MILP with multiple objectives
        
        Args:
            soil_properties: Soil characteristics
            climate_conditions: Climate data
            available_land: Available land in hectares
            budget_limit: Budget constraint in UGX
            objectives: List of optimization objectives
            time_horizon_months: Planning horizon in months
        """
        
        # Get suitable crops
        suitable_crops = self.constraint_engine.get_suitable_crops(soil_properties, climate_conditions)
        if not suitable_crops:
            return self._generate_fallback_plan(soil_properties, climate_conditions, available_land)
        
        crop_names = [crop['crop'] for crop in suitable_crops]
        
        # Create MILP solver
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            print("❌ Could not create MILP solver")
            return None
        
        # Decision variables: x[i] = hectares allocated to crop i
        x = {}
        for i, crop in enumerate(crop_names):
            x[i] = solver.NumVar(0, available_land, f'hectares_{crop}')
        
        # Binary variables for crop selection
        y = {}
        for i, crop in enumerate(crop_names):
            y[i] = solver.BoolVar(f'select_{crop}')
        
        # Constraints
        # 1. Land constraint
        solver.Add(sum(x[i] for i in range(len(crop_names))) <= available_land)
        
        # 2. Minimum allocation constraint (if crop is selected, allocate at least 0.1 ha)
        for i in range(len(crop_names)):
            solver.Add(x[i] >= 0.1 * y[i])
            solver.Add(x[i] <= available_land * y[i])
        
        # 3. Budget constraint (if specified)
        if budget_limit:
            total_cost = sum(
                self.crop_economics[crop_names[i]]['production_cost_per_hectare'] * x[i]
                for i in range(len(crop_names))
            )
            solver.Add(total_cost <= budget_limit)
        
        # 4. Risk diversification constraint (max 60% in any single crop)
        for i in range(len(crop_names)):
            solver.Add(x[i] <= 0.6 * available_land)
        
        # 5. Food security constraint (at least 30% in staple crops)
        staple_crops = ['maize', 'rice', 'cassava', 'sweet_potato']
        staple_indices = [i for i, crop in enumerate(crop_names) if crop in staple_crops]
        if staple_indices:
            solver.Add(sum(x[i] for i in staple_indices) >= 0.3 * available_land)
        
        # Objective function: Weighted combination of multiple objectives
        objective_terms = []
        
        if 'profit' in objectives:
            profit_term = sum(
                (self._calculate_expected_profit(crop_names[i], soil_properties, climate_conditions) * x[i])
                for i in range(len(crop_names))
            )
            objective_terms.append(profit_term)
        
        if 'food_security' in objectives:
            food_security_term = sum(
                (self._calculate_food_security_score(crop_names[i]) * x[i])
                for i in range(len(crop_names))
            )
            objective_terms.append(food_security_term)
        
        if 'risk_diversification' in objectives:
            risk_term = sum(
                (self._calculate_risk_score(crop_names[i]) * x[i])
                for i in range(len(crop_names))
            )
            objective_terms.append(risk_term)
        
        # Combine objectives with weights
        weights = [0.5, 0.3, 0.2]  # profit, food_security, risk_diversification
        combined_objective = sum(w * term for w, term in zip(weights[:len(objective_terms)], objective_terms))
        
        solver.Maximize(combined_objective)
        
        # Solve
        status = solver.Solve()
        
        if status == pywraplp.Solver.OPTIMAL:
            return self._extract_solution(solver, crop_names, x, y, soil_properties, climate_conditions)
        else:
            print(f"❌ Optimization failed with status: {status}")
            return self._generate_fallback_plan(soil_properties, climate_conditions, available_land)
    
    def _calculate_expected_profit(self, crop, soil_properties, climate_conditions):
        """Calculate expected profit per hectare for a crop"""
        if crop not in self.crop_economics:
            return 0
        
        econ = self.crop_economics[crop]
        
        # Get suitability score
        score, _, _ = self.constraint_engine.evaluate_crop_suitability(crop, soil_properties, climate_conditions)
        
        # Calculate expected yield (scaled by suitability)
        min_yield, max_yield = econ['yield_per_hectare']
        expected_yield = min_yield + (max_yield - min_yield) * score
        
        # Calculate expected revenue
        expected_revenue = expected_yield * econ['price_per_ton']
        
        # Calculate profit
        profit = expected_revenue - econ['production_cost_per_hectare']
        
        return profit
    
    def _calculate_food_security_score(self, crop):
        """Calculate food security contribution score"""
        food_security_scores = {
            'maize': 0.9, 'rice': 0.9, 'cassava': 0.8, 'sweet_potato': 0.7,
            'beans': 0.8, 'banana': 0.6, 'groundnut': 0.7,
            'coffee': 0.2, 'cotton': 0.1, 'sugarcane': 0.3
        }
        return food_security_scores.get(crop, 0.5)
    
    def _calculate_risk_score(self, crop):
        """Calculate risk diversification score (higher = lower risk)"""
        risk_scores = {
            'maize': 0.7, 'rice': 0.6, 'cassava': 0.9, 'sweet_potato': 0.8,
            'beans': 0.8, 'banana': 0.6, 'groundnut': 0.7,
            'coffee': 0.4, 'cotton': 0.3, 'sugarcane': 0.5
        }
        return risk_scores.get(crop, 0.5)
    
    def _extract_solution(self, solver, crop_names, x, y, soil_properties, climate_conditions):
        """Extract and format the optimization solution"""
        solution = {
            'allocation': {},
            'total_profit': 0,
            'total_cost': 0,
            'total_land_used': 0,
            'crop_details': [],
            'optimization_summary': {}
        }
        
        for i, crop in enumerate(crop_names):
            hectares = x[i].solution_value()
            if hectares > 0.01:  # Only include crops with meaningful allocation
                solution['allocation'][crop] = hectares
                solution['total_land_used'] += hectares
                
                # Calculate crop-specific metrics
                profit = self._calculate_expected_profit(crop, soil_properties, climate_conditions)
                econ = self.crop_economics[crop]
                
                crop_detail = {
                    'crop': crop,
                    'hectares': hectares,
                    'expected_profit': profit * hectares,
                    'production_cost': econ['production_cost_per_hectare'] * hectares,
                    'expected_yield': self._calculate_expected_yield(crop, soil_properties, climate_conditions) * hectares,
                    'suitability_score': self.constraint_engine.evaluate_crop_suitability(crop, soil_properties, climate_conditions)[0]
                }
                
                solution['crop_details'].append(crop_detail)
                solution['total_profit'] += crop_detail['expected_profit']
                solution['total_cost'] += crop_detail['production_cost']
        
        # Add optimization summary
        solution['optimization_summary'] = {
            'solver_status': 'OPTIMAL',
            'total_crops_selected': len(solution['allocation']),
            'land_utilization': solution['total_land_used'],
            'profit_per_hectare': solution['total_profit'] / max(solution['total_land_used'], 0.001),
            'cost_per_hectare': solution['total_cost'] / max(solution['total_land_used'], 0.001)
        }
        
        return solution
    
    def _calculate_expected_yield(self, crop, soil_properties, climate_conditions):
        """Calculate expected yield per hectare"""
        if crop not in self.crop_economics:
            return 0
        
        score, _, _ = self.constraint_engine.evaluate_crop_suitability(crop, soil_properties, climate_conditions)
        min_yield, max_yield = self.crop_economics[crop]['yield_per_hectare']
        return min_yield + (max_yield - min_yield) * score
    
    def _generate_fallback_plan(self, soil_properties, climate_conditions, available_land):
        """Generate a fallback plan when optimization fails"""
        return {
            'allocation': {'maize': available_land * 0.4, 'beans': available_land * 0.3, 'cassava': available_land * 0.3},
            'total_profit': 0,
            'total_cost': 0,
            'total_land_used': available_land,
            'crop_details': [],
            'optimization_summary': {
                'solver_status': 'FALLBACK',
                'total_crops_selected': 3,
                'land_utilization': available_land,
                'note': 'Using fallback plan due to optimization failure'
            }
        }
    
    def generate_seasonal_plan(self, soil_properties, climate_conditions, 
                             available_land=1.0, planning_months=12):
        """Generate a seasonal crop rotation plan"""
        
        # Get suitable crops
        suitable_crops = self.constraint_engine.get_suitable_crops(soil_properties, climate_conditions)
        crop_names = [crop['crop'] for crop in suitable_crops[:8]]  # Limit to top 8 crops
        
        # Create CP-SAT model for seasonal planning
        model = cp_model.CpModel()
        
        # Convert available_land to integer (representing 0.1 hectare units)
        land_units = int(available_land * 10)  # 1.5 hectares = 15 units of 0.1 ha
        
        # Variables: x[crop, month] = land units planted (0.1 hectare units)
        x = {}
        for crop in crop_names:
            for month in range(1, 13):
                x[crop, month] = model.NewIntVar(0, land_units, f'{crop}_{month}')
        
        # Variables: y[crop, month] = binary selection
        y = {}
        for crop in crop_names:
            for month in range(1, 13):
                y[crop, month] = model.NewBoolVar(f'select_{crop}_{month}')
        
        # Constraints
        # 1. Land constraint per month
        for month in range(1, 13):
            model.Add(sum(x[crop, month] for crop in crop_names) <= land_units)
        
        # 2. Planting season constraints
        for crop in crop_names:
            if crop in self.seasonal_constraints:
                planting_months = self.seasonal_constraints[crop]['planting_months']
                for month in range(1, 13):
                    if month not in planting_months:
                        model.Add(x[crop, month] == 0)
        
        # 3. Crop rotation constraints (avoid same crop in consecutive months)
        for crop in crop_names:
            for month in range(1, 12):
                model.Add(x[crop, month] + x[crop, month + 1] <= land_units)
        
        # 4. Minimum allocation if planted (1 unit = 0.1 hectare)
        for crop in crop_names:
            for month in range(1, 13):
                model.Add(x[crop, month] >= 1 * y[crop, month])  # Minimum 0.1 hectare
                model.Add(x[crop, month] <= land_units * y[crop, month])
        
        # Objective: Maximize total expected profit
        total_profit = 0
        for crop in crop_names:
            for month in range(1, 13):
                profit_per_ha = self._calculate_expected_profit(crop, soil_properties, climate_conditions)
                total_profit += profit_per_ha * x[crop, month]
        
        model.Maximize(total_profit)
        
        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL:
            return self._extract_seasonal_solution(solver, crop_names, x, soil_properties, climate_conditions)
        else:
            print(f"⚠️ CP-SAT optimization failed with status: {status}")
            print("   Falling back to simple seasonal plan")
            return self._generate_simple_seasonal_plan(crop_names, available_land, soil_properties, climate_conditions)
    
    def _extract_seasonal_solution(self, solver, crop_names, x, soil_properties, climate_conditions):
        """Extract seasonal planning solution"""
        seasonal_plan = {
            'monthly_allocations': {},
            'total_profit': 0,
            'crop_rotation_summary': {},
            'monthly_summary': {}
        }
        
        for month in range(1, 13):
            monthly_allocation = {}
            monthly_profit = 0
            
            for crop in crop_names:
                land_units = solver.Value(x[crop, month])
                if land_units > 0:
                    hectares = land_units * 0.1  # Convert back to hectares
                    monthly_allocation[crop] = hectares
                    profit_per_ha = self._calculate_expected_profit(crop, soil_properties, climate_conditions)
                    monthly_profit += profit_per_ha * hectares
            
            seasonal_plan['monthly_allocations'][month] = monthly_allocation
            seasonal_plan['monthly_summary'][month] = {
                'total_hectares': sum(monthly_allocation.values()),
                'total_profit': monthly_profit,
                'crops_planted': len(monthly_allocation)
            }
            seasonal_plan['total_profit'] += monthly_profit
        
        return seasonal_plan
    
    def _generate_simple_seasonal_plan(self, crop_names, available_land, soil_properties, climate_conditions):
        """Generate a simple seasonal plan as fallback"""
        seasonal_plan = {
            'monthly_allocations': {},
            'total_profit': 0,
            'crop_rotation_summary': {},
            'monthly_summary': {}
        }
        
        # Simple rotation: maize -> beans -> cassava -> fallow
        rotation_crops = ['maize', 'beans', 'cassava']
        rotation_crops = [crop for crop in rotation_crops if crop in crop_names]
        
        if not rotation_crops:
            rotation_crops = crop_names[:3]
        
        for month in range(1, 13):
            crop_index = (month - 1) % len(rotation_crops)
            crop = rotation_crops[crop_index]
            
            if crop in self.seasonal_constraints:
                planting_months = self.seasonal_constraints[crop]['planting_months']
                if month in planting_months:
                    seasonal_plan['monthly_allocations'][month] = {crop: available_land}
                    profit_per_ha = self._calculate_expected_profit(crop, soil_properties, climate_conditions)
                    seasonal_plan['total_profit'] += profit_per_ha * available_land
                else:
                    seasonal_plan['monthly_allocations'][month] = {}
            else:
                seasonal_plan['monthly_allocations'][month] = {crop: available_land}
                profit_per_ha = self._calculate_expected_profit(crop, soil_properties, climate_conditions)
                seasonal_plan['total_profit'] += profit_per_ha * available_land
            
            seasonal_plan['monthly_summary'][month] = {
                'total_hectares': sum(seasonal_plan['monthly_allocations'][month].values()),
                'total_profit': 0,
                'crops_planted': len(seasonal_plan['monthly_allocations'][month])
            }
        
        return seasonal_plan

class ComprehensiveEvaluationSystem:
    """
    Comprehensive evaluation system for agricultural recommendations
    """
    
    def __init__(self):
        self.evaluation_metrics = {
            'economic': ['profit', 'cost_efficiency', 'market_accessibility'],
            'environmental': ['soil_health', 'water_efficiency', 'biodiversity'],
            'social': ['food_security', 'labor_intensity', 'gender_equity'],
            'risk': ['climate_resilience', 'market_volatility', 'pest_disease']
        }
        
        print("✅ Comprehensive Evaluation System initialized")
    
    def evaluate_recommendation(self, recommendation, soil_properties, climate_conditions):
        """Evaluate a crop recommendation across multiple dimensions"""
        
        evaluation = {
            'overall_score': 0,
            'dimension_scores': {},
            'detailed_analysis': {},
            'recommendations': []
        }
        
        # Economic evaluation
        economic_score = self._evaluate_economic_dimension(recommendation)
        evaluation['dimension_scores']['economic'] = economic_score
        
        # Environmental evaluation
        environmental_score = self._evaluate_environmental_dimension(recommendation, soil_properties)
        evaluation['dimension_scores']['environmental'] = environmental_score
        
        # Social evaluation
        social_score = self._evaluate_social_dimension(recommendation)
        evaluation['dimension_scores']['social'] = social_score
        
        # Risk evaluation
        risk_score = self._evaluate_risk_dimension(recommendation, climate_conditions)
        evaluation['dimension_scores']['risk'] = risk_score
        
        # Calculate overall score
        weights = {'economic': 0.3, 'environmental': 0.25, 'social': 0.25, 'risk': 0.2}
        evaluation['overall_score'] = sum(
            weights[dim] * score for dim, score in evaluation['dimension_scores'].items()
        )
        
        # Generate recommendations
        evaluation['recommendations'] = self._generate_improvement_recommendations(evaluation)
        
        return evaluation
    
    def _evaluate_economic_dimension(self, recommendation):
        """Evaluate economic aspects"""
        score = 0.5  # Base score
        
        if 'suitable_crops' in recommendation:
            crops = recommendation['suitable_crops']
            if len(crops) >= 3:
                score += 0.2  # Diversification bonus
            
            high_value_crops = ['coffee', 'cotton', 'sugarcane']
            if any(crop['crop'] in high_value_crops for crop in crops):
                score += 0.2  # High-value crop bonus
        
        return min(1.0, score)
    
    def _evaluate_environmental_dimension(self, recommendation, soil_properties):
        """Evaluate environmental aspects"""
        score = 0.5  # Base score
        
        # Soil health consideration
        ph = soil_properties.get('pH', 7.0)
        if 6.0 <= ph <= 7.0:
            score += 0.2
        
        # Crop diversity
        if 'suitable_crops' in recommendation:
            crops = recommendation['suitable_crops']
            if len(crops) >= 4:
                score += 0.2  # Biodiversity bonus
        
        return min(1.0, score)
    
    def _evaluate_social_dimension(self, recommendation):
        """Evaluate social aspects"""
        score = 0.5  # Base score
        
        # Food security crops
        if 'suitable_crops' in recommendation:
            crops = recommendation['suitable_crops']
            staple_crops = ['maize', 'rice', 'cassava', 'sweet_potato', 'beans']
            staple_count = sum(1 for crop in crops if crop['crop'] in staple_crops)
            
            if staple_count >= 2:
                score += 0.3  # Food security bonus
        
        return min(1.0, score)
    
    def _evaluate_risk_dimension(self, recommendation, climate_conditions):
        """Evaluate risk aspects"""
        score = 0.5  # Base score
        
        # Climate resilience
        temp = climate_conditions.get('temperature_mean', 25)
        rainfall = climate_conditions.get('rainfall_mean', 800)
        
        if 20 <= temp <= 30 and 500 <= rainfall <= 1500:
            score += 0.3  # Favorable climate
        
        # Crop diversification
        if 'suitable_crops' in recommendation:
            crops = recommendation['suitable_crops']
            if len(crops) >= 3:
                score += 0.2  # Risk diversification bonus
        
        return min(1.0, score)
    
    def _generate_improvement_recommendations(self, evaluation):
        """Generate recommendations for improvement"""
        recommendations = []
        
        for dimension, score in evaluation['dimension_scores'].items():
            if score < 0.6:
                if dimension == 'economic':
                    recommendations.append("Consider including high-value crops like coffee or cotton")
                elif dimension == 'environmental':
                    recommendations.append("Implement soil improvement practices and crop rotation")
                elif dimension == 'social':
                    recommendations.append("Include more staple food crops for food security")
                elif dimension == 'risk':
                    recommendations.append("Diversify crops to reduce climate and market risks")
        
        return recommendations

# Initialize the multi-objective cropping planner
print("\nInitializing Multi-Objective Cropping Planner...")

# Create planner instance
planner = MultiObjectiveCroppingPlanner(
    constraint_engine=constraint_engine,
    semantic_retriever=semantic_retriever
)

# Create evaluation system
evaluation_system = ComprehensiveEvaluationSystem()

print("✅ Multi-Objective Cropping Planner and Evaluation System initialized!")

# Test the optimization system
print("\nTesting Multi-Objective Optimization...")

# Test parameters
test_soil = {
    'pH': 6.2,
    'organic_matter': 2.1,
    'texture_class': 'loamy'
}

test_climate = {
    'temperature_mean': 24,
    'rainfall_mean': 750,
    'humidity_mean': 65
}

# Test 1: Basic optimization
print("\n1. Testing Basic Crop Allocation Optimization...")
basic_plan = planner.optimize_crop_allocation(
    soil_properties=test_soil,
    climate_conditions=test_climate,
    available_land=2.0,
    budget_limit=3000000,  # 3M UGX
    objectives=['profit', 'food_security', 'risk_diversification']
)

if basic_plan:
    print(f"✅ Basic optimization completed!")
    print(f"   Total crops selected: {basic_plan['optimization_summary']['total_crops_selected']}")
    print(f"   Total profit: {basic_plan['total_profit']:,.0f} UGX")
    print(f"   Land utilization: {basic_plan['total_land_used']:.2f} hectares")
    print(f"   Profit per hectare: {basic_plan['optimization_summary']['profit_per_hectare']:,.0f} UGX")

# Test 2: Seasonal planning
print("\n2. Testing Seasonal Crop Rotation Planning...")
seasonal_plan = planner.generate_seasonal_plan(
    soil_properties=test_soil,
    climate_conditions=test_climate,
    available_land=1.5,
    planning_months=12
)

if seasonal_plan:
    print(f"✅ Seasonal planning completed!")
    print(f"   Total expected profit: {seasonal_plan['total_profit']:,.0f} UGX")
    print(f"   Planning horizon: 12 months")

# Test 3: Comprehensive evaluation
print("\n3. Testing Comprehensive Evaluation System...")
test_recommendation = constrained_rag.generate_constrained_recommendation(test_soil, test_climate)
evaluation = evaluation_system.evaluate_recommendation(test_recommendation, test_soil, test_climate)

print(f"✅ Evaluation completed!")
print(f"   Overall score: {evaluation['overall_score']:.2f}")
print(f"   Economic score: {evaluation['dimension_scores']['economic']:.2f}")
print(f"   Environmental score: {evaluation['dimension_scores']['environmental']:.2f}")
print(f"   Social score: {evaluation['dimension_scores']['social']:.2f}")
print(f"   Risk score: {evaluation['dimension_scores']['risk']:.2f}")

if evaluation['recommendations']:
    print("   Improvement recommendations:")
    for rec in evaluation['recommendations']:
        print(f"     • {rec}")

print("\n" + "="*70)
print("MULTI-OBJECTIVE CROPPING PLANNER COMPLETE")
print("="*70)
print("Next steps:")
print("1. Build recommendation API with Flask/FastAPI")
print("2. Create web interface for farmers")
print("3. Implement real-time data integration")
print("4. Deploy production system")
print("5. Performance monitoring and optimization")
