import os
import time
import shutil
import hashlib
import requests
import re
from dataclasses import dataclass, field
from git import Repo
from pathlib import Path
from xml.dom import minidom
import xml.etree.ElementTree as ET
import xml.etree.ElementTree as etree
from datetime import datetime, timedelta
from typing import Union, Dict, Any, List, Iterable, Optional, Tuple
from .analysis import Analysis, count_java_methods_in_file

# =========================
# Print utilities
# =========================

def printWarning(message):
    print("\033[93m WARNING: " + message + "\033[0m")


def printError(message):
    print("\033[91m ERROR: " + message + "\033[0m")


def printInfo(message):
    print("\033[96m INFO: " + message + "\033[0m")


# =========================
# Configuration models
# =========================

@dataclass
class Settings:
    git_url: str = ""
    local_path: str = ""
    language: str = "java"  # "java" or "c"
    clone_detector_tool: Optional[str] = None  # "nicad" | "simian" | None

    # Temporal scope / commit selection
    from_begin: bool = False
    specific_commit: str = ""  # when provided, uses SPECIFIC..HEAD as range
    use_merge_commits: bool = False
    use_days: bool = False
    days: Optional[int] = None
    use_leaps: bool = False
    commit_leaps: Optional[int] = None


@dataclass
class Paths:
    script_dir: str = "scripts"
    tools_dir: str = "tools"
    res_dir: str = "final_results"
    cur_res_dir: str = field(default_factory=lambda: os.path.join("final_results", "0000000"))
    # NOTE: ws_dir/repo_dir/data_dir/prod_data_dir/hist_file are reassigned in main()
    ws_dir: str = "workspace"  # legacy default; overwritten in main()
    repo_dir: str = field(default_factory=lambda: os.path.join("workspace", "repo"))  # overwritten in main()
    data_dir: str = field(default_factory=lambda: os.path.join("workspace", "dataset"))  # overwritten in main()
    prod_data_dir: str = field(default_factory=lambda: os.path.join("workspace", "dataset", "production"))  # overwritten in main()
    clone_detector_dir: str = "clone_detector_result"
    clone_detector_xml: str = field(default_factory=lambda: os.path.join("clone_detector_result", "result.xml"))

    hist_file: str = field(default_factory=lambda: os.path.join("workspace", "githistory.txt"))  # overwritten in main()
    p_res_file: str = field(default_factory=lambda: os.path.join("final_results", "production_results.xml"))
    p_dens_file: str = field(default_factory=lambda: os.path.join("final_results", "production_density.csv"))


@dataclass
class State:
    # Data accumulated in memory
    p_lin_data: List["Lineage"] = field(default_factory=list)
    p_dens_data: List[Tuple[int, float, float]] = field(default_factory=list)


@dataclass
class Context:
    settings: Settings
    paths: Paths
    state: State


# =========================
# Domain classes
# =========================

class CloneFragment:
    def __init__(self, file, ls, le, fh=0):
        # replace /dataset/production with /repo to keep compatibility with the original pipeline
        self.file = file.replace("/dataset/production", "/repo")
        self.ls = ls
        self.le = le
        self.function_hash = fh
        self.hash = hashlib.sha256(f"{self.file}{self.ls}{self.le}".encode("utf-8")).hexdigest()[:7]

    def contains(self, other):
        return self.file == other.file and self.ls <= other.ls and self.le >= other.le

    def __eq__(self, other):
        return self.file == other.file and self.ls == other.ls and self.le == other.le

    def matches(self, other):
        return self.hash == other.hash

    def matchesStrictly(self, other):
        return self.file == other.file and (self.ls == other.ls or self.function_hash == other.function_hash)

    def __hash__(self):
        return hash(self.file + str(self.ls))

    def toXML(self):
        return '\t\t\t<source file="%s" startline="%d" endline="%d" hash="%d"></source>\n' % (self.file, self.ls, self.le, self.function_hash)

    def countLOC(self):
        return self.le - self.ls


class CloneClass:
    def __init__(self):
        self.fragments: List[CloneFragment] = []

    def contains(self, fragment: CloneFragment):
        return any(f.matches(fragment) for f in self.fragments)

    def matches(self, cc: "CloneClass"):
        n = 0
        for fragment in cc.fragments:
            if self.contains(fragment):
                n += 1
        return (n == len(cc.fragments)) or (n == len(self.fragments))

    def toXML(self):
        s = '\t\t<class nclones="%d">\n' % (len(self.fragments))
        for fragment in self.fragments:
            try:
                s += fragment.toXML()
            except Exception:
                pass
        s += "\t\t</class>\n"
        return s

    def countLOC(self):
        return sum(f.countLOC() for f in self.fragments)


class CloneVersion:
    def __init__(self, cc=None, h=None, n=None, evo="None", chan="None"):
        self.cloneclass = cc
        self.hash = h
        self.parent_hash = ""
        self.nr = n
        self.evolution_pattern = evo
        self.change_pattern = chan
        self.removed_fragments: List[CloneFragment] = []

    def toXMLRemoved(self):
        s = ""
        for f in self.removed_fragments:
            s += f.toXML()
        return s

    def toXML(self):
        s = '\t<version nr="%d" hash="%s" evolution="%s" change="%s" parent_hash="%s">\n' % (
            self.nr,
            self.hash,
            self.evolution_pattern,
            self.change_pattern,
            self.parent_hash,
        )
        try:
            s += self.cloneclass.toXML()
        except Exception:
            pass
        s += "\t</version>\n"
        if self.removed_fragments:
            s += self.toXMLRemoved()
        return s


class Lineage:
    def __init__(self):
        self.versions: List[CloneVersion] = []

    def matches(self, cc: CloneClass):
        # matches if any fragment of the cloneclass exists in the last version
        for fragment in cc.fragments:
            if self.versions[-1].cloneclass.contains(fragment):
                return True
        return False

    def toXML(self):
        s = "<lineage>\n"
        for version in self.versions:
            s += version.toXML()
        s += "</lineage>\n"
        return s


# =========================
# Auxiliary parsers
# =========================

def getLastCommitFromDensityCSV(filename):
    try:
        with open(filename, "r+", encoding="utf-8") as file:
            return int(file.readlines()[-1].split(",")[0])
    except Exception as e:
        printError("Something went wrong while parsing the density dataset:")
        printError(str(e))
        return 0


def getDataFromCSV(filename):
    try:
        with open(filename, "r+", encoding="utf-8") as file:
            data = []
            for line in file.readlines():
                data.append((int(line.split(",")[0]), float(line.split(",")[1]), float(line.split(",")[2])))
            return data
    except Exception as e:
        printError("Something went wrong while parsing the CSV file:")
        printError(str(e))
        return 0


def parseLineageFile(filename):
    lineages = []
    with open(filename, "r+", encoding="utf-8") as file:
        try:
            file_xml = etree.parse(file)
            for lineage in file_xml.getroot():
                lin = Lineage()
                versions = list(lineage)
                for version in versions:
                    cc = CloneClass()
                    cloneclasses = list(version)
                    if len(cloneclasses) != 1:
                        printWarning("Unexpected amount of clone classes in version.")
                        printWarning("Please check if input file is consistent.")
                    for fragment in list(cloneclasses[0]):
                        cc.fragments.append(
                            CloneFragment(
                                fragment.get("file"),
                                int(fragment.get("startline")),
                                int(fragment.get("endline")),
                                fragment.get("function"),
                                int(fragment.get("hash")),
                            )
                        )
                    cv = CloneVersion(
                        cc,
                        version.get("hash"),
                        int(version.get("nr")),
                        version.get("evolution"),
                        version.get("change"),
                    )
                    lin.versions.append(cv)
                lineages.append(lin)
        except Exception as e:
            printError("Something went wrong while parsing the lineage dataset:")
            printError(str(e))
            return []
    return lineages


def _read_text_with_fallback(path):
    with open(path, "rb") as f:
        data = f.read()
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def GetCloneFragment(filename, startline, endline):
    text = _read_text_with_fallback(filename)
    lines_out = [""]
    remove_ws = False
    for i, raw in enumerate(text.splitlines(True)):
        if i >= (startline - 1):
            base = raw.split("//", 1)[0]
            if base.strip() == "":
                continue
            l = base.rstrip()
            if remove_ws:
                l = l.lstrip()
                remove_ws = False
            if len(l) > 2 and l[-1] not in ";{":
                remove_ws = True
            else:
                l = l + "\n"
            lines_out.append(l)
        if i >= (endline - 1):
            break
    return "".join(lines_out)


def GetPattern(v1: CloneVersion, v2: CloneVersion):
    evolution = "None"
    if len(v1.cloneclass.fragments) == len(v2.cloneclass.fragments):
        evolution = "Same"
    elif len(v1.cloneclass.fragments) > len(v2.cloneclass.fragments):
        evolution = "Subtract"
    else:
        evolution = "Add"

    def matches_count(a: Iterable[CloneFragment], b: Iterable[CloneFragment]):
        n = 0
        for f2 in b:
            for f1 in a:
                if f1.function_hash == f2.function_hash:
                    n += 1
                    break
        return n

    change = "None"
    if evolution in ("Same", "Subtract"):
        nr_of_matches = matches_count(v1.cloneclass.fragments, v2.cloneclass.fragments)
        if nr_of_matches == len(v2.cloneclass.fragments):
            change = "Same"
        elif nr_of_matches == 0:
            change = "Consistent"
        else:
            change = "Inconsistent"
    elif evolution == "Add":
        nr_of_matches = matches_count(v2.cloneclass.fragments, v1.cloneclass.fragments)
        if nr_of_matches == len(v1.cloneclass.fragments):
            change = "Same"
        elif nr_of_matches == 0:
            change = "Consistent"
        else:
            change = "Inconsistent"

    return (evolution, change)


# =========================
# Pipeline (no globals)
# =========================
def SetupRepo(ctx: Context):
    s, p = ctx.settings, ctx.paths

    # Local path: copy as-is
    if s.local_path:
        src = Path(s.local_path)
        dest = Path(p.repo_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        return

    print("Setting up local directory for git repository " + s.git_url)

    repo_git_dir = os.path.join(p.repo_dir, ".git")

    if os.path.isdir(repo_git_dir):
        # Always fetch first
        os.system(f'git -C "{p.repo_dir}" fetch --all --prune')

        # Discover current branch (skip if detached)
        head = os.popen(f'git -C "{p.repo_dir}" rev-parse --abbrev-ref HEAD').read().strip()
        if head and head != "HEAD":
            # Check if upstream is configured without using @{u}
            remote = os.popen(f'git -C "{p.repo_dir}" config --get branch.{head}.remote').read().strip()
            merge  = os.popen(f'git -C "{p.repo_dir}" config --get branch.{head}.merge').read().strip()

            if not remote or not merge:
                # If there's an 'origin' remote, set upstream to origin/<branch>
                has_origin = (os.system(f'git -C "{p.repo_dir}" remote get-url origin >NUL 2>&1') == 0) \
                             or (os.system(f'git -C "{p.repo_dir}" remote get-url origin >/dev/null 2>&1') == 0)
                if has_origin:
                    os.system(f'git -C "{p.repo_dir}" branch --set-upstream-to "origin/{head}" "{head}" >NUL 2>&1 || true')
                    os.system(f'git -C "{p.repo_dir}" branch --set-upstream-to "origin/{head}" "{head}" >/dev/null 2>&1 || true')

            # Fast-forward only; ignore non-FF errors
            os.system(f'git -C "{p.repo_dir}" pull --ff-only >NUL 2>&1 || true')
            os.system(f'git -C "{p.repo_dir}" pull --ff-only >/dev/null 2>&1 || true')
        else:
            printInfo("Detached HEAD or no branch; fetched refs only.")
        return

    # Not a git repo but folder exists → clean it
    if os.path.isdir(p.repo_dir):
        shutil.rmtree(p.repo_dir, ignore_errors=True)

    # Clone fresh
    os.makedirs(p.ws_dir, exist_ok=True)
    os.system(f'git clone "{s.git_url}" "{p.repo_dir}"')
    print(" Repository setup complete.\n")


def PrepareGitHistory(ctx: Context):
    print("Getting git history")
    s, p = ctx.settings, ctx.paths
    now = datetime.now()
    repo = Repo(p.repo_dir)

    rev = "HEAD"
    iter_kwargs: Dict[str, Any] = {}

    if s.specific_commit:
        rev = f"{s.specific_commit}..HEAD"
    elif s.use_days and s.days is not None:
        iter_kwargs = {"since": (now - timedelta(days=int(s.days or 0))).strftime('%Y-%m-%d %H:%M:%S')}
    elif s.from_begin:
        rev = "--all"

    if s.use_merge_commits:
        iter_kwargs["merges"] = True

    commits = repo.iter_commits(rev, **iter_kwargs)

    if s.use_merge_commits:
        commits = (c for c in commits if len(c.parents) > 1)

    if s.use_leaps and int(s.commit_leaps or 0) > 1:
        step = int(s.commit_leaps)
        commits = (c for i, c in enumerate(commits) if i % step == 0)

    commits = list(commits)
    lines = [f"{c.hexsha[:7]} {datetime.fromtimestamp(c.committed_date).date()} {c.author.name} {c.summary}" for c in commits]
    Path(p.hist_file).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(lines)} commit(s) to {p.hist_file}")


def GetHashes(ctx: Context) -> List[str]:
    p = ctx.paths
    hashes: List[str] = []
    if not os.path.exists(p.hist_file):
        return hashes
    with open(p.hist_file, "rb") as fp:
        for raw in fp:
            raw = raw.strip()
            if not raw:
                continue
            first = raw.split(None, 1)[0]
            try:
                h = first.decode("ascii")
            except UnicodeDecodeError:
                continue
            hashes.append(h)
    hashes.reverse()
    return hashes

def PrepareSourceCode(ctx: Context) -> bool:
    s, p = ctx.settings, ctx.paths
    print("Preparing source code")
    found = False

    repo_root = os.path.abspath(p.repo_dir)
    if not os.path.isdir(repo_root):
        printError(f"Repository directory not found: {repo_root}")
        return False

    # Reset output dirs
    if os.path.exists(p.data_dir):
        shutil.rmtree(p.data_dir, ignore_errors=True)
    os.makedirs(p.res_dir, exist_ok=True)
    os.makedirs(p.clone_detector_dir, exist_ok=True)
    os.makedirs(p.data_dir, exist_ok=True)
    os.makedirs(p.prod_data_dir, exist_ok=True)

    lang = (s.language or "").lower().strip().lstrip(".")  # e.g., "java" or "c"
    if not lang:
        printError("Language not set.")
        return False

    repo_path = Path(repo_root)

    # Recursively pick files by extension; skip .git and files with "test" in the filename
    for src in repo_path.rglob(f"*.{lang}"):
        # Skip anything inside .git
        if any(part == ".git" for part in src.parts):
            continue
        # Skip files whose name contains "test" (case-insensitive)
        if "test" in src.name.lower():
            continue

        # Relative dir of the source; root files go directly into production/
        rel_dir = os.path.relpath(str(src.parent), repo_root)
        dst_dir = p.prod_data_dir if rel_dir == "." else os.path.join(p.prod_data_dir, rel_dir)

        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(str(src), os.path.join(dst_dir, src.name))
        found = True

    print("Source code ready for clone analysis.\n")
    return found


def StartFromPreviousVersion(ctx: Context) -> int:
    p = ctx.paths
    if not os.path.exists(p.res_dir):
        os.makedirs(p.res_dir, exist_ok=True)
        return 0
    return 0


def parse_clones_xml(xml_input: Union[str, bytes]) -> Dict[str, Any]:
    if isinstance(xml_input, (bytes,)):
        root = ET.fromstring(xml_input)
    elif isinstance(xml_input, str) and (xml_input.strip().startswith("<") or "\n" in xml_input):
        root = ET.fromstring(xml_input)
    else:
        if not os.path.exists(xml_input):
            raise FileNotFoundError(f"XML file not found: {xml_input}")
        tree = ET.parse(xml_input)
        root = tree.getroot()

    if root.tag != "clones":
        raise ValueError(f"Unexpected root tag '{root.tag}', expected 'clones'.")

    result: Dict[str, List[Dict[str, Any]]] = {"clones": []}
    for class_el in root.findall("class"):
        class_dict = {"sources": []}
        for src in class_el.findall("source"):
            file_path = src.get("file")
            startline = int(src.get("startline")) if src.get("startline") is not None else None
            endline = int(src.get("endline")) if src.get("endline") is not None else None
            class_dict["sources"].append({"file": file_path, "startline": startline, "endline": endline})
        result["clones"].append(class_dict)
    return result


def parse_simian_to_clones(simian_xml: str) -> None:
    raw = open(simian_xml, 'r', encoding='utf-8', errors='ignore').read()
    pos = raw.find("<simian")
    if pos == -1:
        pos = raw.find("<")
        if pos == -1:
            raise ValueError("Content does not appear to contain valid XML.")
    xml = raw[pos:]
    xml = re.sub(r"<!--.*?-->", "", xml, flags=re.DOTALL)
    xml = re.sub(r"<\?.*?\?>", "", xml, flags=re.DOTALL)

    clean_xml = xml.strip()
    root = ET.fromstring(clean_xml)
    check = root.find("check")
    if check is None:
        raise ValueError("Invalid XML: <check> node not found.")

    clones = ET.Element("clones")
    for set_node in check.findall("set"):
        class_el = ET.SubElement(clones, "class")
        for block in set_node.findall("block"):
            source_file = block.get("sourceFile")
            start = block.get("startLineNumber")
            end = block.get("endLineNumber")
            ET.SubElement(class_el, "source", {"file": source_file, "startline": start or "", "endline": end or ""})

    rough = ET.tostring(clones, encoding="utf-8")
    reparsed = minidom.parseString(rough)
    result_xml = reparsed.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")
    with open(simian_xml, "w", encoding="utf-8") as f:
        f.write(result_xml)

def find_method_end(lines, decl_line, brace_col):
    depth = 0
    for li in range(decl_line - 1, len(lines)):
        line = lines[li].split("//", 1)[0]
        start_c = brace_col if li == decl_line - 1 else 0
        for cj in range(start_c, len(line)):
            ch = line[cj]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return li + 1
    return None


def RunCloneDetection(ctx: Context):
    s, p = ctx.settings, ctx.paths
    print("Starting clone detection:")

    clone_detector_result = Path(p.clone_detector_dir)
    clone_detector_result.mkdir(parents=True, exist_ok=True)
    for item in clone_detector_result.iterdir():
        if item.is_file():
            item.unlink()

    tool = (s.clone_detector_tool or "").casefold()
    if tool == "nicad":
        print(" >>> Running nicad6...")
        os.makedirs(p.cur_res_dir, exist_ok=True)

        nicad_dir = os.path.join(p.tools_dir, "NiCad")                 # absolute
        target_dir = p.prod_data_dir                                   # absolute

        # run NiCad with absolute paths
        os.system(f'cd "{nicad_dir}" && ./nicad6 functions {s.language} "{target_dir}"')

        # NiCad outputs to "<target_dir>_functions-clones/production_functions-clones-0.30-classes.xml"
        nicad_xml_dir = f'{target_dir}_functions-clones'
        nicad_xml = os.path.join(nicad_xml_dir, "production_functions-clones-0.30-classes.xml")

        # move result into our fixed detector output location
        shutil.move(nicad_xml, p.clone_detector_xml)

        # cleanup NiCad temp/output folders and logs
        os.system(f'rm -rf "{nicad_xml_dir}"')
        os.system(f'rm "{ctx.paths.data_dir}"/*.log 2>/dev/null || true')

        # NO path rewrite needed anymore (we used absolutes); keep /dataset/production
        # so CloneFragment can map it to /repo.
        return

    if tool == "simian":
        print(" >>> Running Simian...")
        java_jar_command = f'java -jar {p.tools_dir}/simian/simian-4.0.0.jar'  # absolute
        options_command = "-formatter=xml"
        simian_command = f'{java_jar_command} {options_command} "{p.prod_data_dir}"/*.{s.language} > "{p.clone_detector_xml}"'
        os.system(simian_command)
        parse_simian_to_clones(p.clone_detector_xml)
        return

    # If no tool is selected, just clean the folder
    for f in Path(p.clone_detector_dir).iterdir():
        if f.is_file():
            f.unlink()
    print(" Finished clone detection.\n")


def parseCloneClassFile(ctx: Context, cloneclass_filename: str) -> List[CloneClass]:
    print(cloneclass_filename)
    cloneclasses: List[CloneClass] = []
    try:
        file_xml = ET.parse(cloneclass_filename)
        root = file_xml.getroot()
        for child in root:
            cc = CloneClass()
            fragments = list(child)
            if not fragments:
                continue
            for fragment in fragments:
                file_path = fragment.get("file")
                startline = int(fragment.get("startline"))
                endline = int(fragment.get("endline"))
                cf = CloneFragment(file_path, startline, endline)
                cf.function_hash = hash(GetCloneFragment(cf.file, cf.ls, cf.le))
                cc.fragments.append(cf)
            cloneclasses.append(cc)
    except Exception as e:
        printError("Something went wrong while parsing the clonepair dataset:")
        raise e
    return cloneclasses


def CheckDoubleMatch(cc_original: CloneClass, cc1: CloneClass, cc2: CloneClass) -> int:
    cc1_strict_match = False
    cc2_strict_match = False
    for fragment in cc_original.fragments:
        for f1 in cc1.fragments:
            if fragment.matchesStrictly(f1):
                cc1_strict_match = True
        for f2 in cc2.fragments:
            if fragment.matchesStrictly(f2):
                cc2_strict_match = True
    if cc1_strict_match == cc2_strict_match:
        return 0
    if cc1_strict_match:
        return 1
    elif cc2_strict_match:
        return 2
    return 0


def RunDensityAnalysis(ctx: Context, commitNr: int, pcloneclasses: List[CloneClass]):
    s, p, st = ctx.settings, ctx.paths, ctx.state
    print("Starting density analysis:")
    print(" > Production code...")
    if not os.path.exists(p.clone_detector_xml):
        st.p_dens_data.append((commitNr, 0, 0))
        return

    current_clones = parseCloneClassFile(ctx, p.clone_detector_xml)
    if len(current_clones) == 0:
        st.p_dens_data.append((commitNr, 0, 0))
        return

    all_paths = set()
    for clone in current_clones:
        for fr in clone.fragments:
            all_paths.add(fr.file)

    total_amount_of_p_functions = sum([count_java_methods_in_file(path) for path in all_paths])

    all_sources: List[CloneFragment] = []
    for clone in current_clones:
        all_sources.extend(clone.fragments)

    amount_of_cloned_p_functions = len(all_sources)
    try:
        density_f_p = 100 * (float(amount_of_cloned_p_functions) / total_amount_of_p_functions)
    except Exception:
        density_f_p = 0.0

    cloc_p_out = (
        os.popen(f'perl {ctx.paths.tools_dir}/cloc/cloc-1.72.pl {ctx.paths.prod_data_dir}/ | grep "SUM"').read().split()
    )
    if len(cloc_p_out) >= 3:
        total_amount_of_p_loc = int(cloc_p_out[-1]) + int(cloc_p_out[-2]) + int(cloc_p_out[-3])
    else:
        total_amount_of_p_loc = 0

    amount_of_cloned_p_loc = sum(cc.countLOC() for cc in pcloneclasses)
    density_loc_p = 100 * (float(amount_of_cloned_p_loc) / total_amount_of_p_loc) if total_amount_of_p_loc else 0.0

    st.p_dens_data.append((commitNr, density_f_p, density_loc_p))
    print(" Finished density analysis.\n")


def RunGenealogyAnalysis(ctx: Context, commitNr: int, hash_: str):
    s, p, st = ctx.settings, ctx.paths, ctx.state
    print("Starting genealogy analysis:")
    print(" > Production code...")
    pcloneclasses = parseCloneClassFile(ctx, p.clone_detector_xml)

    if not st.p_lin_data:
        for pcc in pcloneclasses:
            v = CloneVersion(pcc, hash_, commitNr)
            l = Lineage()
            l.versions.append(v)
            st.p_lin_data.append(l)
    else:
        for pcc in pcloneclasses:
            found = False
            for lineage in st.p_lin_data:
                if lineage.matches(pcc):
                    if lineage.versions[-1].nr == commitNr:
                        if len(lineage.versions) < 2:
                            continue
                        checkDoubleMatch = CheckDoubleMatch(
                            lineage.versions[-2].cloneclass,
                            lineage.versions[-1].cloneclass,
                            pcc,
                        )
                        if checkDoubleMatch == 1:
                            continue
                        elif checkDoubleMatch == 2:
                            pcloneclasses.append(lineage.versions[-1].cloneclass)
                            lineage.versions.pop()

                    evolution, change = GetPattern(lineage.versions[-1], CloneVersion(pcc))
                    if (
                        evolution == "Same"
                        and change == "Same"
                        and lineage.versions[-1].evolution_pattern == "Same"
                        and lineage.versions[-1].change_pattern == "Same"
                    ):
                        lineage.versions[-1].nr = commitNr
                        lineage.versions[-1].hash = hash_
                    else:
                        lineage.versions.append(CloneVersion(pcc, hash_, commitNr, evolution, change))
                    found = True
                    break
            if not found:
                v = CloneVersion(pcc, hash_, commitNr)
                l = Lineage()
                l.versions.append(v)
                st.p_lin_data.append(l)

    print(" Finished genealogy analysis.\n")
    RunDensityAnalysis(ctx, commitNr, pcloneclasses)

def build_no_clones_message(detector: Optional[str]) -> str:
    detector_name = (detector or "unspecified").strip() or "unspecified"

    root = ET.Element("result")
    ET.SubElement(root, "status").text = "no_clones_found"
    ET.SubElement(root, "detector").text = detector_name
    ET.SubElement(root, "message").text = (
        f"No clones were found across the analyzed commits using the '{detector_name}' detector."
    )

    # Try modern pretty print (Python 3.9+)
    try:
        ET.indent(root, space="  ", level=0)  # type: ignore[attr-defined]
        return ET.tostring(root, encoding="utf-8", xml_declaration=True).decode("utf-8")
    except Exception:
        # Fallback: use minidom for pretty printing
        rough = ET.tostring(root, encoding="utf-8")
        reparsed = minidom.parseString(rough)
        return reparsed.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")

def build_genealogy_xml(lineages_xml: str, metrics_xml: str) -> str:
    def _normalize_whitespace(elem):
        if elem.text is not None and elem.text.strip() == "":
            elem.text = None
        if elem.tail is not None and elem.tail.strip() == "":
            elem.tail = None
        for c in list(elem):
            _normalize_whitespace(c)

    lineages_root = ET.fromstring(lineages_xml)
    metrics_root = ET.fromstring(metrics_xml)

    _normalize_whitespace(lineages_root)
    _normalize_whitespace(metrics_root)

    genealogy = ET.Element("genealogy")
    genealogy.append(lineages_root)
    genealogy.append(metrics_root)

    try:
        ET.indent(genealogy, space="  ", level=0)  # Python 3.9+
        return ET.tostring(
            genealogy,
            encoding="utf-8",
            xml_declaration=True,
            short_empty_elements=False,  # force </genealogy>
        ).decode("utf-8")
    except AttributeError:
        rough = ET.tostring(
            genealogy,
            encoding="utf-8",
            xml_declaration=True,
            short_empty_elements=False,  # same in fallback
        )
        dom = minidom.parseString(rough)
        return dom.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")


def WriteLineageFile(ctx: Context, lineages: List[Lineage], filename: str):
    xml_txt = "<lineages>\n"

    with open(filename, "w+", encoding="utf-8") as output_file:
        output_file.write("<lineages>\n")
        for lineage in lineages:
            output_file.write(lineage.toXML())
            xml_txt += lineage.toXML()
        output_file.write("</lineages>\n")
        xml_txt += "</lineages>\n"

    return xml_txt

def WriteDensityFile(ctx: Context, densitys: List[Tuple[int, float, float]], filename: str):
    with open(filename, "w+", encoding="utf-8") as output_file:
        for density in densitys:
            output_file.write(f"{density[0]}, {density[1]}, {density[2]}\n")


def timeToString(seconds):
    result = ""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = (seconds % 3600) % 60
    if hours:
        result += str(hours) + " hours, "
    if minutes:
        result += str(minutes) + " minutes, "
    result += str(seconds) + " seconds"
    return result


def insert_parent_hash(ctx: Context, parent_hash: str):
    for lineage in ctx.state.p_lin_data:
        lineage.versions[-1].parent_hash = parent_hash


# =========================
# Settings initialization from user dictionary
# =========================

def init_settings_from_user(general_settings: Dict[str, Any]) -> Settings:
    user = general_settings.get("user_settings", {}) or {}

    # correct field names
    git_repository = general_settings.get("git_repository", "")
    local_path = general_settings.get("local_path", "")
    language = user.get("language") or general_settings.get("language", "java")

    from_first_commit = bool(user.get("from_first_commit"))
    specific_commit = user.get("from_a_specific_commit") or ""
    days_prior = user.get("days_prior")  # may be None or int

    merge_commit = user.get("merge_commit")  # optional
    fixed_leaps = user.get("fixed_leaps")    # optional

    clone_detector = user.get("clone_detector")  # REQUIRED by your rule

    s = Settings(
        git_url=git_repository,
        local_path=local_path,
        language=language,
        clone_detector_tool=clone_detector,

        from_begin=from_first_commit,
        specific_commit=specific_commit or "",
        use_days=days_prior is not None,
        days=days_prior,

        use_merge_commits=bool(merge_commit),
        use_leaps=bool(fixed_leaps),
        commit_leaps=fixed_leaps,
    )
    return s


def _derive_repo_name(settings: Settings) -> str:
    """
    Determine a stable repository folder name from git_url or local_path.
    Falls back to 'repo' if nothing can be inferred.
    """
    if settings.local_path:
        return os.path.basename(os.path.normpath(settings.local_path)) or "repo"
    url = (settings.git_url or "").rstrip("/")
    base = os.path.basename(url) or "repo"
    if base.endswith(".git"):
        base = base[:-4]
    base = os.path.splitext(base)[0] or base
    return base or "repo"


# =========================
# Main function (no globals)
# =========================

def validate_user_input_or_raise(general_settings: Dict[str, Any]) -> None:
    """
    Rules:
    - Required: git_repository (root) and clone_detector (inside user_settings)
    - Select exactly ONE among:
        * from_first_commit (bool True)
        * from_a_specific_commit (non-empty string)
        * days_prior (int > 0)
    - merge_commit and fixed_leaps are optional.
    """
    if not isinstance(general_settings, dict):
        raise ValueError("Invalid configuration: root object must be a dictionary.")

    user = general_settings.get("user_settings", {}) or {}

    # Required
    git_repo = general_settings.get("git_repository")
    if not git_repo or not isinstance(git_repo, str) or not git_repo.strip():
        raise ValueError("Missing required field: 'git_repository'.")

    clone_detector = user.get("clone_detector")
    if not clone_detector or not isinstance(clone_detector, str) or not clone_detector.strip():
        raise ValueError("Missing required field in 'user_settings': 'clone_detector'.")

    # Exclusivity among the three selectors
    ffc = bool(user.get("from_first_commit"))
    fac = user.get("from_a_specific_commit")
    fac_ok = isinstance(fac, str) and fac.strip() != ""
    dp = user.get("days_prior")
    dp_ok = isinstance(dp, int) and dp > 0

    chosen = sum([1 if ffc else 0, 1 if fac_ok else 0, 1 if dp_ok else 0])

    if chosen == 0:
        raise ValueError(
            "You must provide exactly ONE of "
            "'from_first_commit', 'from_a_specific_commit', or 'days_prior'."
        )
    if chosen > 1:
        raise ValueError(
            "Invalid configuration: provide only ONE of "
            "'from_first_commit', 'from_a_specific_commit', or 'days_prior' (mutually exclusive)."
        )

    # If days_prior is present but invalid (0, negative, non-int)
    if dp is not None and not dp_ok:
        raise ValueError("'days_prior' must be an integer > 0 when provided.")

    # Others (merge_commit, fixed_leaps) are optional; no strict validation here.


def execute_omniccg(general_settings: Dict[str, Any]) -> str:
    validate_user_input_or_raise(general_settings)
    settings = init_settings_from_user(general_settings)
    paths = Paths()
    state = State()
    ctx = Context(settings=settings, paths=paths, state=state)

    # --- NEW: make all folders live inside the installed package directory ---
    pkg_root = Path(__file__).resolve().parent            # .../omniccg
    pkg_root_str = str(pkg_root)

    # Tools / scripts
    paths.tools_dir = os.path.join(pkg_root_str, "tools")
    paths.script_dir = os.path.join(pkg_root_str, "scripts")

    # Results & detector output
    paths.res_dir = os.path.join(pkg_root_str, "results")
    paths.cur_res_dir = os.path.join(paths.res_dir, "0000000")
    paths.clone_detector_dir = os.path.join(pkg_root_str, "clone_detector_result")
    paths.clone_detector_xml = os.path.join(paths.clone_detector_dir, "result.xml")

    # Output files
    paths.p_res_file = os.path.join(paths.res_dir, "production_results.xml")
    paths.p_dens_file = os.path.join(paths.res_dir, "production_density.csv")

    # Workspace (clones, datasets, history) lives under omniccg/cloned_repositories/<repo_name>
    repo_name = _derive_repo_name(settings)
    base_dir = os.path.join(pkg_root_str, "cloned_repositories", repo_name)
    paths.ws_dir = base_dir
    paths.repo_dir = os.path.join(base_dir, "repo")
    paths.data_dir = os.path.join(base_dir, "dataset")
    paths.prod_data_dir = os.path.join(paths.data_dir, "production")
    paths.hist_file = os.path.join(base_dir, "githistory.txt")

    # Ensure folders exist
    os.makedirs(paths.res_dir, exist_ok=True)
    os.makedirs(paths.clone_detector_dir, exist_ok=True)
    os.makedirs(base_dir, exist_ok=True)
    # --- 

    # Cleanup previous results
    if os.path.isdir(paths.res_dir):
        shutil.rmtree(paths.res_dir)

    print("STARTING DATA COLLECTION SCRIPT\n")
    SetupRepo(ctx)
    PrepareGitHistory(ctx)
    hashes = GetHashes(ctx)
    time.sleep(1)

    analysis_index = 0
    total_time = 0

    for hash_index in range(len(hashes)):
        iteration_start_time = time.time()
        analysis_index += 1
        current_hash = hashes[hash_index]
        hi_plus = hash_index + 1

        printInfo("Analyzing commit nr." + str(hi_plus) + " with hash " + current_hash)
        paths.cur_res_dir = os.path.join(paths.res_dir, f"{hi_plus}_{current_hash}")

        # Ensure we are at the correct commit
        show_out = os.popen(f"git --git-dir {paths.repo_dir}/.git show --oneline -s").read()
        if current_hash not in show_out:
            os.system(f"(cd {paths.repo_dir}; git checkout {current_hash} -f > /dev/null 2>&1)")
            time.sleep(1)

        # Prepare source and run detection
        if not PrepareSourceCode(ctx):
            continue

        RunCloneDetection(ctx)
        RunGenealogyAnalysis(ctx, hi_plus, current_hash)
        WriteLineageFile(ctx, ctx.state.p_lin_data, paths.p_res_file)

        if hi_plus != 1:
            insert_parent_hash(ctx, hashes[hi_plus - 1])

        # Cleanup
        try:
            shutil.rmtree(paths.cur_res_dir, ignore_errors=True)
        except Exception:
            pass

        # Timing
        iteration_end_time = time.time()
        iteration_time = iteration_end_time - iteration_start_time
        total_time += iteration_time

        print("Iteration finished in " + timeToString(int(iteration_time)))
        avg = int(total_time / analysis_index) if analysis_index else 0
        remaining = int((total_time / analysis_index) * (len(hashes) - analysis_index)) if analysis_index else 0
        print(" >>> Average iteration time: " + timeToString(avg))
        print(" >>> Estimated remaining time: " + timeToString(remaining))

        WriteLineageFile(ctx, ctx.state.p_lin_data, paths.p_res_file)
        time.sleep(1)

    # If nothing was accumulated, return a clear XML message
    if len(ctx.state.p_lin_data) == 0:
        return build_no_clones_message(settings.clone_detector_tool)

    # Otherwise, finalize outputs
    WriteDensityFile(ctx, ctx.state.p_dens_data, paths.p_dens_file)
    lineages_xml = WriteLineageFile(ctx, ctx.state.p_lin_data, paths.p_res_file)
    metrics_xml = Analysis(paths.res_dir)
    genealogy_xml = build_genealogy_xml(lineages_xml, metrics_xml)

    print("\nDONE")
    return genealogy_xml
