MAX_LENGTH_BY_TASK = {
    'count': 47,
    'int-sum': 15,
    'dyck': 37
}

MODEL_NAME_BY_TASK = {
    'count': 'count-model.dill',
    'int-sum': '3-sum-model.dill',
    'dyck': 'dyck-model.dill'
}

OOD_EVALS_BY_TASK = {
    'count': ['x_only', 'replace_y', 'length_ood'],
    'int-sum': ['cascading_overflow', 'decimals', 'length_ood'],
    'dyck': ['almost_balanced', 'ood_new_token', 'length_ood']
}


