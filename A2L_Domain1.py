"""Domain1 entry for the shared A2L training implementation."""
import json
import os

from A2L import REGRESSION_LAGGED, load_runtime_dependencies, main
from Utils.runtime import effective_config, parse_domain_args

POLICY = REGRESSION_LAGGED


if __name__ == "__main__":
    args = parse_domain_args("Domain1")
    if args.print_config:
        print(json.dumps(effective_config(args), indent=2))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        load_runtime_dependencies()
        main(args, policy=POLICY)
