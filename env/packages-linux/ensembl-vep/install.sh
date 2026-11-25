#!/usr/bin/env bash
# Ensembl VEP via conda-forge/bioconda at /opt/ensembl-vep (vep on PATH). The
# species cache is large task DATA, not installed here.
set -euo pipefail
TARGET_HOME="$(getent passwd 1000 | cut -d: -f6)"
[ -n "$TARGET_HOME" ] || { echo "[pkg ensembl-vep] FATAL: uid 1000 has no passwd entry" >&2; exit 1; }
MM="$TARGET_HOME/.local/bin/micromamba"
[ -x "$MM" ] || { echo "[pkg ensembl-vep] FATAL: micromamba required first" >&2; exit 1; }
P=/opt/ensembl-vep
export MAMBA_ROOT_PREFIX="$TARGET_HOME/.local/share/micromamba"
if [ ! -x "$P/bin/vep" ]; then
  "$MM" create -y -p "$P" -c bioconda -c conda-forge ensembl-vep
  chown -R 1000:0 "$P" 2>/dev/null || true
fi
test -x "$P/bin/vep" || { echo "[pkg ensembl-vep] FATAL: vep missing" >&2; exit 1; }
# vep is a `#!/usr/bin/env perl` script; a bare symlink runs it under SYSTEM perl
# (no DBI) -> "Can't locate DBI.pm". Wrap it so the conda env's perl (which has DBI
# + the EnsEMBL libs) is first on PATH.
rm -f /usr/local/bin/vep   # may be an old symlink; cat> would write through it
cat > /usr/local/bin/vep <<EOF
#!/usr/bin/env bash
export PATH="$P/bin:\$PATH"
exec "$P/bin/vep" "\$@"
EOF
chmod +x /usr/local/bin/vep
vep --help 2>&1 | grep -qi "variant effect predictor" || { echo "[pkg ensembl-vep] FATAL: vep did not run" >&2; exit 1; }
echo "[pkg ensembl-vep] OK"
