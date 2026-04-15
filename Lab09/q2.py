from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
bn = DiscreteBayesianNetwork([
    ('Intelligence', 'Grade'),
    ('StudyHours', 'Grade'),
    ('Difficulty', 'Grade'),
    ('Grade', 'Pass')
])

cpd_I = TabularCPD(variable='Intelligence', variable_card=2, values=[[0.7], [0.3]])
cpd_S = TabularCPD(variable='StudyHours', variable_card=2, values=[[0.6], [0.4]])
cpd_D = TabularCPD(variable='Difficulty', variable_card=2, values=[[0.4], [0.6]])

cpd_G = TabularCPD(
    variable='Grade', variable_card=3,
    values=[
        [0.8, 0.6, 0.5, 0.3, 0.4, 0.2, 0.2, 0.1],
        [0.15, 0.3, 0.3, 0.4, 0.4, 0.5, 0.4, 0.3],
        [0.05, 0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.6]
    ],
    evidence=['Intelligence', 'StudyHours', 'Difficulty'],
    evidence_card=[2, 2, 2]
)

cpd_P = TabularCPD(
    variable='Pass', variable_card=2,
    values=[
        [0.05, 0.20, 0.50],  # Fail
        [0.95, 0.80, 0.50]   # Pass
    ],
    evidence=['Grade'],
    evidence_card=[3]
)

bn.add_cpds(cpd_I, cpd_S, cpd_D, cpd_G, cpd_P)
assert bn.check_model()
inference_engine = VariableElimination(bn)

# Query 1: P(Pass | StudyHours=Sufficient(0), Difficulty=Hard(0))
result1 = inference_engine.query(
    variables=['Pass'],
    evidence={'StudyHours': 0, 'Difficulty': 0}
)

print("P(Pass | StudyHours=Sufficient, Difficulty=Hard):")
print(result1)

# Query 2: P(Intelligence=High(0) | Pass=Yes(1))
result2 = inference_engine.query(
    variables=['Intelligence'],
    evidence={'Pass': 1}
)

print("\nP(Intelligence | Pass=Yes):")
print(result2)
