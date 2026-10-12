import argparse
from . import workflow


def main():
    stages=['metadata','knowledge','compile','embed','prepare','train','predict','fit-cells','generate','evaluate',
            'fit-update-oof','fit-update','update','evaluate-update','prepare-explain','explain','score-explain']
    parser=argparse.ArgumentParser(description='Predict cellular responses, explain evidence, and update predictions from measurements.')
    parser.add_argument('stage',choices=stages,help='Workflow stage to run.');parser.add_argument('--config',required=True,help='Path to the JSON configuration file.')
    args=parser.parse_args();config=workflow.configure(args.config)
    from . import cells,update,evaluate
    actions={'metadata':workflow.prepare_metadata,'knowledge':workflow.prepare_knowledge,'compile':workflow.compile_evidence,
        'embed':workflow.embed,'prepare':workflow.prepare_model,'train':lambda c:workflow.training.fit(c['run_root']/'model'),
        'predict':workflow.predict,'fit-cells':cells.fit,'generate':cells.generate,'evaluate':evaluate.cells,
        'fit-update-oof':update.fit_oof,'fit-update':update.fit,'update':update.run,'evaluate-update':evaluate.update,
        'prepare-explain':workflow.prepare_explain,'explain':workflow.explain,'score-explain':workflow.score_explain}
    actions[args.stage](config)


if __name__=='__main__':main()
