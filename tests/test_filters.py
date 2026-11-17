import pytest

from jobbot.filters import RuleFilter, required_years
from jobbot.text import fold

from .conftest import make_config, make_job


@pytest.fixture
def rules(cfg):
    return RuleFilter(cfg)


@pytest.mark.parametrize("title", ["Senior Data Engineer", "Sr. Python Developer", "Lead Data Scientist",
                                   "Engineering Manager, Backend", "Staff Software Engineer"])
def test_seniority_titles_dropped(rules, title):
    v = rules.prefilter(make_job(title=title))
    assert not v.ok and "title has" in v.reason


def test_staffing_is_not_staff(rules):
    # whole-word matching: "Staffing" must not trigger "staff"
    assert "staff" not in rules.prefilter(make_job(title="Staffing Data Engineer")).reason


def test_role_keyword_required(rules):
    assert not rules.prefilter(make_job(title="Graphic Designer")).ok
    assert rules.prefilter(make_job(title="Junior Data Engineer")).ok
    assert rules.prefilter(make_job(title="Python Developer")).ok
    # "etl" must be a whole word, not part of "settlement"
    assert not rules.prefilter(make_job(title="Settlement Analyst")).ok


def test_location_rules_and_priority(rules):
    assert rules.prefilter(make_job(location="Cluj-Napoca")).priority == "high"
    assert rules.prefilter(make_job(location="Bucharest - Dorobanti, Romania; Cluj, Romania")).priority == "high"
    v = rules.prefilter(make_job(location="București, Romania"))
    assert v.ok and v.priority == "low" and v.location_rule.name == "Bucharest"
    assert not rules.prefilter(make_job(location="London, United Kingdom")).ok


def test_romania_needs_remote(rules):
    assert rules.prefilter(make_job(location="Remote, Romania")).location_rule.name == "Romania (remote)"
    assert not rules.prefilter(make_job(location="Iasi, Romania")).ok
    # country-only location: tentatively kept until the description is known...
    job = make_job(location="Romania", description=None)
    assert rules.prefilter(job).ok
    # ...then decided by the description
    job.description = "This is a distributed role, work from anywhere in the country."
    assert rules.check(job).ok
    job.description = "You will work from our office."
    assert not rules.check(job).ok


def test_years_of_experience(rules):
    assert not rules.check(make_job(description="Requirements: 3+ years of experience with Python")).ok
    assert not rules.check(make_job(description="Minim 3 ani de experiență în dezvoltare")).ok
    assert not rules.check(make_job(description="At least five years experience in data")).ok
    assert rules.check(make_job(description="1-2 years of experience with SQL")).ok
    assert rules.check(make_job(description="Our company has over 20 years of experience")).ok


@pytest.mark.parametrize("text,expected", [
    ("3+ years of professional experience", 3),
    ("2-4 years of experience", 2),
    ("minimum 4 yrs experience", 4),
    ("experience: 5 years", 5),
    ("1 year of experience; 5 years experience is nice to have", 1),
    ("founded 30 years ago, with 30 years of experience", None),
    ("no experience required", None),
    # years + in/of/as + a field, without the word "experience" (seen in a PwC AI Engineer posting)
    ("what we need from you\n\n· 3+ years in software, data or AI engineering.", 3),
    ("at least 4 years of Python development", 4),
    ("3 years working in data engineering", 3),
    ("2+ years’ professional work in backend teams", 2),
    ("5 years as a data engineer", 5),
    ("minim 3 ani în domeniul IT", 3),
    ("cel puțin 4 ani de lucru în dezvoltare software", 4),
    ("3 ani ca programator Python", 3),
    # ...but not company history, durations or contracts
    ("For 30+ years, we've delivered impact; for 5 years in a row we won awards", None),
    ("we have been in this business for 10 years in Cluj", None),
    ("contract: 1 year contract, with the possibility of part time working", None),
    ("Our team has 8 years in fintech", None),
    ("acum 3 ani în urmă am deschis biroul din Cluj", None),
    ("a 2 year program of the company", None),
    ("you need to have spent at least 1 year in your current role", 1),
])
def test_required_years(text, expected):
    assert required_years(fold(text)) == expected


def test_unpaid_internship_dropped(rules):
    job = make_job(title="Data Engineer Intern", description="This is an unpaid internship for students.")
    assert not rules.check(job).ok
    job = make_job(title="Data Engineer Intern", description="Stagiu neplătit de 3 luni.")
    assert not rules.check(job).ok
    assert rules.check(make_job(title="Data Engineer Intern", description="Paid internship.")).ok


@pytest.mark.parametrize("description,why", [
    ("- Fluent in English and German (any additional language is a plus).", "requires German"),
    ("- Advanced German & English language skills, both written and spoken", "requires German"),
    ("- You are fluent in English and French", "requires French"),
    ("Cerințe:\n- Cunoașterea limbii germane la nivel avansat", "requires German"),
    ("- Limba franceză - nivel C1", "requires French"),
    ("- Native Dutch speaker", "requires Dutch"),
    ("- You will graduate in 2027 with a Master's or PhD degree in a quantitative field", "Master's/PhD only"),
    ("- You are a Master’s student with practical ML experience", "Master's/PhD only"),
    ("Bursă dedicată studenților masteranzi.", "Master's/PhD only"),
    ("- Hybrid role requiring 3 days per week in our London, UK office and 2 days remote.", "3+ on-site days"),
    ("Program hibrid: 2 zile pe săptămână la biroul din Berlin.", "2+ on-site days"),
])
def test_hard_requirements_drop(rules, description, why):
    v = rules.check(make_job(title="Data Engineer Intern", description=description))
    assert not v.ok and v.reason.startswith(why), v.reason


@pytest.mark.parametrize("description", [
    "Fluent English. German is a plus.",
    "Fluent English; knowledge of French would be an advantage",
    "Limba germană constituie un avantaj.",
    "Fluent in English or German",
    "We are a German company with offices in Cluj.",
    "Bachelor's or Master's degree in Computer Science",
    "Eligibility:\n- Bachelor's degree program, graduating in 2027\n- Master's or Ph.D. degree program, graduating 2028",
    "A Master's degree is a plus.",
    "Studii superioare (licență sau master).",
    "Hybrid arrangement: 3 days onsite at the office, 2 days remote work from home",
    "Hybrid: 3 days a week in our Cluj or London office.",
    "Up to 2 days per year in our London office for team events.",
    "Teams meet two to four times yearly in person, in London or Lisbon.",
])
def test_hard_requirements_keep(rules, description):
    v = rules.check(make_job(title="Data Engineer Intern", description=description))
    assert v.ok, v.reason


def test_hard_requirements_configurable(tmp_path):
    base = make_config(tmp_path).profile.model_dump()
    base["hard_requirements"] = {"languages": {"French": ["french"]}, "onsite_abroad": {"min_days": 4}}
    rules = RuleFilter(make_config(tmp_path, profile=base))
    german = make_job(description="- Fluent in English and German")
    assert rules.check(german).ok                     # German is no longer listed
    assert not rules.check(make_job(description="- Fluent French required")).ok
    london = make_job(description="Requires 3 days per week in our London office.")
    assert rules.check(london).ok                     # below min_days


def test_entry_level_keywords(tmp_path):
    base = make_config(tmp_path).profile.model_dump()
    base["entry_level_keywords"] = ["intern", "internship", "working student", "entry-level", "stagiar", "junior"]
    rules = RuleFilter(make_config(tmp_path, profile=base))
    for title in ["Working Student - Finance", "Internship in AI", "Stagiar IT (m/f)", "Entry-Level Analyst",
                  "Junior Consultant"]:
        assert rules.prefilter(make_job(title=title)).role == "Entry level", title
    assert rules.prefilter(make_job(title="Data Engineer Intern")).role == "Data Engineer"  # target role wins
    assert not rules.prefilter(make_job(title="Internal Auditor")).ok                       # whole words only
    assert not rules.prefilter(make_job(title="Senior Intern Mentor")).ok                   # exclusions still apply
    assert not RuleFilter(make_config(tmp_path)).prefilter(make_job(title="Working Student - Finance")).ok


def test_entry_level_keywords_stay_out_of_prompt(tmp_path):
    from jobbot.scoring.base import build_system_prompt
    base = make_config(tmp_path).profile.model_dump()
    plain = build_system_prompt(make_config(tmp_path), "CV")
    base["entry_level_keywords"] = ["intern", "stagiar"]
    assert build_system_prompt(make_config(tmp_path, profile=base), "CV") == plain


def test_exclude_title_at_employer(tmp_path):
    base = make_config(tmp_path).profile.model_dump()
    base["entry_level_keywords"] = ["working student"]
    base["exclude_title_at_employer"] = [{"employer": "Contoso", "title_contains": ["working student"]}]
    rules = RuleFilter(make_config(tmp_path, profile=base))
    v = rules.prefilter(make_job(title="Working Student Production IT", company="Contoso Industries GmbH"))
    assert not v.ok and v.reason == "title has 'working student' at Contoso"
    assert rules.prefilter(make_job(title="Working Student Production IT", company="Endava")).ok
    assert rules.prefilter(make_job(title="Junior Data Engineer", company="Contoso")).ok
    assert make_config(tmp_path).profile.exclude_title_at_employer == []   # off by default
