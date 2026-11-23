#!/usr/bin/env bash
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANTF_G1_ROOT="$ROOT" source "$ROOT/scripts/paths.sh"
ISAAC_DIR="${ISAAC_DIR:-$HOME/isaacsim}"
MAP="$HG1_ROOT/ros2_ws/src/hospital-sim-ros2/g1_navigation/maps/hospital_gt.yaml"
CFG="$HOME/.config/mantf-g1-hospital"
mkdir -p "$CFG"

cat > "$CFG/env.sh" <<SH
export MANTF_ROOT="$MANTF_ROOT"
export HG1_ROOT="$HG1_ROOT"
export MANTF_G1_ROOT="$ROOT"
export MANTF_G1_MAP="$MAP"
export ISAAC_DIR="$ISAAC_DIR"
export ISAACSIM_PATH="$ISAAC_DIR"
export ISAACSIM_PYTHON_EXE="$ISAAC_DIR/python.sh"
export ISAAC_ASSET_ROOT_60="\$HOME/isaacsim_assets/Assets/Isaac/6.0"
export ROS_DOMAIN_ID="\${ROS_DOMAIN_ID:-2}"
export MANTF_RMW="\${MANTF_RMW:-rmw_cyclonedds_cpp}"
mg1() {
  case "\$1" in
    cd) cd "\$MANTF_G1_ROOT" ;;
    shell) pixi shell --manifest-path "\$MANTF_G1_ROOT/pixi.toml" ;;
    wardzero) source "\$HG1_ROOT/env.sh" ;;
    ""|-h|--help) echo "mg1 <stack|scene|robot|case|goto|status|stop|offline|pickplace|fleet|people|viewer|rviz|test|isaac|isaac-smoke|build|shell|cd|wardzero> [args]" ;;
    *) pixi run --manifest-path "\$MANTF_G1_ROOT/pixi.toml" "\$@" ;;
  esac
}
SH

cat > "$CFG/env.fish" <<FISH
set -gx MANTF_ROOT "$MANTF_ROOT"
set -gx HG1_ROOT "$HG1_ROOT"
set -gx MANTF_G1_ROOT "$ROOT"
set -gx MANTF_G1_MAP "$MAP"
set -gx ISAAC_DIR "$ISAAC_DIR"
set -gx ISAACSIM_PATH "$ISAAC_DIR"
set -gx ISAACSIM_PYTHON_EXE "$ISAAC_DIR/python.sh"
set -gx ISAAC_ASSET_ROOT_60 "\$HOME/isaacsim_assets/Assets/Isaac/6.0"
set -q ROS_DOMAIN_ID; or set -gx ROS_DOMAIN_ID 2
set -q MANTF_RMW; or set -gx MANTF_RMW rmw_cyclonedds_cpp
function mg1
    switch "\$argv[1]"
        case cd
            cd \$MANTF_G1_ROOT
        case shell
            pixi shell --manifest-path \$MANTF_G1_ROOT/pixi.toml
        case wardzero
            bash --rcfile (echo "source ~/.bashrc; source \$HG1_ROOT/env.sh" | psub) -i
        case '' -h --help
            echo "mg1 <stack|scene|robot|case|goto|status|stop|offline|pickplace|fleet|people|viewer|rviz|test|isaac|isaac-smoke|build|shell|cd|wardzero> [args]"
        case '*'
            pixi run --manifest-path \$MANTF_G1_ROOT/pixi.toml \$argv
    end
end
FISH

add_block() {
  local file="$1" line="$2"
  touch "$file"
  if ! grep -q ">>> mantf-g1-hospital >>>" "$file"; then
    cp "$file" "$file.bak.mantf-g1"
    printf '\n# >>> mantf-g1-hospital >>>\n%s\n# <<< mantf-g1-hospital <<<\n' "$line" >> "$file"
    echo "added to $file (backup $file.bak.mantf-g1)"
  else
    echo "already in $file"
  fi
}
add_block "$HOME/.bashrc" "[ -f \"$CFG/env.sh\" ] && source \"$CFG/env.sh\""
mkdir -p "$HOME/.config/fish"
add_block "$HOME/.config/fish/config.fish" "test -f \"$CFG/env.fish\"; and source \"$CFG/env.fish\""
echo "open a new shell (bash or fish), then: mg1 --help"
