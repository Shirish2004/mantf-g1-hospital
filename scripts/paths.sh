# Where the two dependencies are (sourced by every script). Order: the environment variable if set, then the git
# submodule inside this repository if it is checked out, then a sibling folder next to this repository.
#   mantf    (the model, unchanged):         $MANTF_ROOT  | ./mantf     | ../mantf
#   WardZero (hospital, G1, simulators):     $HG1_ROOT    | ./WardZero  | ../WardZero
_mg1_root="${MANTF_G1_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [ -z "${MANTF_ROOT:-}" ]; then
  if [ -f "$_mg1_root/mantf/models/model_igibson_n.py" ]; then MANTF_ROOT="$_mg1_root/mantf"
  else MANTF_ROOT="$(cd "$_mg1_root/../mantf" 2>/dev/null && pwd)"; fi
fi
if [ -z "${HG1_ROOT:-}" ]; then
  if [ -f "$_mg1_root/WardZero/env.sh" ]; then HG1_ROOT="$_mg1_root/WardZero"
  else HG1_ROOT="$(cd "$_mg1_root/../WardZero" 2>/dev/null && pwd)"; fi
fi
export MANTF_ROOT HG1_ROOT
unset _mg1_root
