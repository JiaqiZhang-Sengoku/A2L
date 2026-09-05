"""Domain4 entry: joint Domain2/Domain1 adaptation and compound/open evaluation."""
import json
import os

from A2L import CLASSIFICATION, load_runtime_dependencies, main
from Utils.runtime import effective_config, parse_domain_args

POLICY = CLASSIFICATION


if __name__ == "__main__":
    args = parse_domain_args("Domain4")
    if args.print_config:
        print(json.dumps(effective_config(args), indent=2))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        load_runtime_dependencies()
        main(args, policy=POLICY)
