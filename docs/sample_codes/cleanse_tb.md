---
hide:
  - navigation
---
# **Sample Code 1**
[Return to Sample Codes](../sample_codes.md)
## **Text-based Analysis of Novel Dataset**
This Python script is a comprehensive text analysis tool designed for processing and analyzing congressional hearing transcripts. It combines multiple natural language processing techniques, including sentiment analysis, OCR error correction, speaker identification, and committee member matching. The script utilizes various libraries such as spaCy, pandas, and the Hugging Face transformers library to handle tasks ranging from basic text preprocessing to advanced entity recognition. Key features include the ability to parse complex document structures, match speakers with committee members, correct OCR errors, and analyze sentiment in congressional discussions.

## **Code**
```python
import re
import spacy
import numpy as np
import pandas as pd

from os.path import join, basename
from glob import glob
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

from transformers import pipeline

pipe = pipeline(
    "text-classification",
    model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
    max_length=512,
    truncation=True,
)

root = "C:/Users/lihou/Box/[Redacted]"
metadata = join(root, "data", "ProQuest", "processed_metadata")

# maybe this is a better idea
chairman_ocr_errors = [
    "The CHAIRMAN",
    "The CIIAIRMM",
    "The CHAIRMAx",
    "The CHAIRMINAN",
    # Omitted
]

# -------------------------------------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------------------------------------


def get_correct_txt_from_folder(folder, us_states):
    """
    Identifies the text file in a folder that most likely contains a name list by analyzing state references.
    
    Args:
        folder (str): Path to the folder containing text files
        us_states (list): List of US state names/abbreviations to check for
    
    Returns:
        str: Name of the file with highest state-to-line ratio, or None if no files found
    """
    txt_files = glob(join(folder, "*.txt"))
    if not txt_files:
        return None

    result_files = {}
    
    for txt_file in txt_files:
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            content = ''.join(lines)
            line_count = len(lines)
            
            if line_count == 0:
                continue
                
            state_count = sum(content.count(state) for state in us_states)
            file_name = basename(txt_file)
            result_files[file_name] = state_count / line_count

    return max(result_files, key=result_files.get) if result_files else None


def get_start_indices(txt):
    """
    Identifies starting indices for main committee and subcommittees in a text file.
    
    Args:
        txt (str): Path to the text file to analyze
    
    Returns:
        tuple: (main_committee_index, subcommittee_indices_dict)
            - main_committee_index (int): Line index where main committee starts
            - subcommittee_indices_dict (dict): Dictionary mapping subcommittee numbers to their starting line indices
    """
    main_start = None
    subcommittee_starts = {}
    subcommittee_count = 0

    with open(txt, "r") as f:
        for i, line in enumerate(f):
            words = line.split()
            
            # Skip empty lines
            if not words:
                continue
                
            # Check for main committee
            if main_start is None and any(fuzz.ratio(word, "COMMITTEE") > 70 for word in words):
                main_start = i
                continue
                
            # Check for subcommittees (only after main committee is found)
            if main_start is not None and any(fuzz.ratio(word, "SUBCOMMITTEE") > 70 for word in words):
                subcommittee_starts[subcommittee_count] = i
                subcommittee_count += 1

    return main_start, subcommittee_starts


def parse_line(line, suffix_list, us_states, identity_list):
    """
    Parse a line containing speaker information in various formats and return a DataFrame.
    
    Handles the following formats:
    1. Normal: 'SPEAKER, State' or 'SPEAKER, Identity'
    2. Both: 'SPEAKER, State, Identity'
    3. Line change errors with mixed capitalization
    4. Suffix variations
    
    Args:
        line (str): Input line to parse
        suffix_list (list): List of valid suffixes
        us_states (list): List of US state names/abbreviations
        identity_list (list): List of known identities
    
    Returns:
        pd.DataFrame: DataFrame with columns [Speaker, State, Identity, Suffix, Original Line]
    """
    dataframe = pd.DataFrame(columns=["Speaker", "State", "Identity", "Suffix", "Original Line"])
    updated_identity_list = identity_list.copy()
    
    # Clean and prepare the line
    line = line.replace("\\n", "")
    updated_lines = split_line_on_pattern(line)
    
    # Process each line after potential splitting
    for updated_line in updated_lines:
        # Parse parts and handle suffix
        parts = [part.strip() for part in updated_line.split(",")]
        parts, suffix = extract_suffix(parts, suffix_list)
        
        if not is_valid_name_length(parts):
            continue
            
        # Process based on number of parts
        if len(parts) == 1:
            row = create_single_part_row(parts[0], suffix, line)
        elif len(parts) == 2:
            row = create_double_part_row(parts[0], parts[1], suffix, line, us_states, updated_identity_list)
        elif len(parts) == 3:
            row = create_triple_part_row(parts, suffix, line, us_states, updated_identity_list)
        else:
            continue
            
        if row is not None:
            dataframe = pd.concat([dataframe, pd.DataFrame([row])], ignore_index=True)
            
    return dataframe

def split_line_on_pattern(line):
    """Split line on pattern indicating missing newlines."""
    pattern = re.compile(r"([A-Z][a-z]+) ([A-Z][A-Z]+)")
    parts = line.split(", ")
    updated_lines = []
    
    for i, part in enumerate(parts):
        match = pattern.search(part)
        if match:
            new_parts = pattern.sub(r"\1\n\2", part).split("\n")
            if i > 0:
                updated_lines.append(f"{', '.join(parts[:i])}, {new_parts[0]}")
            else:
                updated_lines.append(new_parts[0])
                
            if i < len(parts) - 1:
                updated_lines.append(f"{new_parts[1]}, {', '.join(parts[i + 1:])}")
            else:
                updated_lines.append(new_parts[1])
            return updated_lines
            
    return [line]

def extract_suffix(parts, suffix_list):
    """Extract suffix from parts if present."""
    suffix = None
    parts_no_suffix = []
    
    for part in parts:
        if part in suffix_list:
            suffix = part
        else:
            parts_no_suffix.append(part)
            
    return parts_no_suffix, suffix

def is_valid_name_length(parts):
    """Check if name length is valid."""
    return not (parts and len(parts[0].split()) > 5)

def create_single_part_row(speaker, suffix, original_line):
    """Create row for single part entry."""
    return {
        "Speaker": speaker,
        "State": None,
        "Identity": None,
        "Suffix": suffix,
        "Original Line": original_line
    }

def create_double_part_row(speaker, second_part, suffix, original_line, us_states, identity_list):
    """Create row for double part entry."""
    state = process.extractOne(second_part, us_states, score_cutoff=65)
    
    if state is not None:
        return {
            "Speaker": speaker,
            "State": state[0],
            "Identity": None,
            "Suffix": suffix,
            "Original Line": original_line
        }
    else:
        if not any(second_part in identity for identity in identity_list):
            identity_list.append(second_part)
        return {
            "Speaker": speaker,
            "State": None,
            "Identity": second_part,
            "Suffix": suffix,
            "Original Line": original_line
        }

def create_triple_part_row(parts, suffix, original_line, us_states, identity_list):
    """Create row for triple part entry."""
    speaker, part2, part3 = parts
    state_part2 = process.extractOne(part2, us_states, score_cutoff=65)
    state_part3 = process.extractOne(part3, us_states, score_cutoff=65)
    
    if state_part2 is not None and state_part3 is None:
        if not any(part3 in identity for identity in identity_list):
            identity_list.append(part3)
        return {
            "Speaker": speaker,
            "State": state_part2[0],
            "Identity": part3,
            "Suffix": suffix,
            "Original Line": original_line
        }
    elif state_part3 is not None and state_part2 is None:
        if not any(part2 in identity for identity in identity_list):
            identity_list.append(part2)
        return {
            "Speaker": speaker,
            "State": state_part3[0],
            "Identity": part2,
            "Suffix": suffix,
            "Original Line": original_line
        }
    return None


def generate_ranges(main_index, sub_indices, total_lines):
    """
    This function serves to generate line indices based on the main_index and sub_indices.
    """
    subs = []
    if main_index is None:
        return []

    # Filter sub_indices to remove main_index
    else:
        for i in range(len(sub_indices)):
            if sub_indices[i] != main_index:
                subs.append(sub_indices[i])

    if len(subs) == 0:
        return [(main_index, total_lines)]
    else:
        subs.sort()
        # return (main_index, subs[0]), (subs[0], subs[1]), ..., (subs[-1], total_lines)
        ranges = [(main_index, subs[0])]
        for i in range(len(subs) - 1):
            ranges.append((subs[i], subs[i + 1]))
        ranges.append((subs[-1], total_lines))
        return ranges


def extract_committee(txt, suffix_list, us_states, identity_list):
    """
    Extract potential speakers from committee and subcommittee sections in a text file.
    
    Args:
        txt (str): Path to the text file to analyze.
        suffix_list (list): List of valid suffixes.
        us_states (list): List of US state names/abbreviations.
        identity_list (list): List of known identities.
    
    Returns:
        pd.DataFrame: DataFrame containing speaker information.
    """
    file_name = basename(txt)
    year = int(re.findall(r"\d{4}", file_name)[0])

    dataframe = pd.DataFrame(
        columns=[
            "Speaker", "State", "Identity", "Suffix", "Original Line",
            "Committee", "File Name", "Year"
        ]
    )
    main_index, sub_indices = get_start_indices(txt)

    with open(txt, "r") as f:
        lines = f.readlines()
        total_lines = len(lines)
        ranges = generate_ranges(main_index, sub_indices, total_lines)

        if not ranges:
            return dataframe

        for start, end in ranges:
            end = min(end, start + 50)
            committee = None

            for i, line in enumerate(lines[start:end], start=start):
                if i == start:
                    committee = line.lower()
                else:
                    df_to_concat = parse_line(line, suffix_list, us_states, identity_list)
                    df_to_concat["Committee"] = committee
                    df_to_concat["File Name"] = file_name
                    df_to_concat["Year"] = year
                    dataframe = pd.concat([dataframe, df_to_concat], ignore_index=True)

    return dataframe



def correct_identity(identity, acceptable_identity_list):
    """
    Corrects the identity of a speaker based on an acceptable identity list.
    
    Args:
        identity (str): The identity to be corrected.
        acceptable_identity_list (list): List of acceptable identities.
    
    Returns:
        str: Corrected identity or 'no_identity'/'no_match' if not found.
    """
    if not identity or identity in [None, ".", ""]:
        return "no_identity"
    
    identity = identity.lower()
    result = process.extractOne(identity, acceptable_identity_list, score_cutoff=70)
    
    return result[0] if result else "no_match"



def estimate_individual_probability(name, entity_type, nlp):
    templates = [
        f"During the hearing, {name} provided expert testimony on the matter.",
        f"{name}, a recognized authority in the field, was cited frequently during the discussions.",
        f"The statement by {name} was referenced multiple times by committee members.",
        f"Senator Johnson addressed {name} directly during the questioning session.",
        f"As noted in the proceedings, {name} has been a key figure in this investigation.",
        f"{name}'s written testimony, submitted prior to the session, outlined several key points.",
        f"The amendment proposed by {name} was considered during the debate.",
        f"Congresswoman Smith thanked {name} for their detailed report on the issue.",
        f"The chairperson asked {name} to clarify their previous statements.",
        f"According to {name}, the data presented during the hearing was conclusive.",
        f"In his closing remarks, the attorney general mentioned {name}'s contributions to the case.",
        f"The recent policy paper by {name} was mentioned as a significant piece of evidence.",
        f"During the panel discussion, {name} argued for stricter regulations.",
        f"The insights from {name} were pivotal to the committee's understanding of the issue.",
        f"After the session, {name} was interviewed by several media outlets regarding their stance.",
        f"{name} was among the experts invited to discuss the implications of the new legislation.",
        f"The findings from {name}'s research were heavily debated during the hearing.",
        f"In an unexpected move, {name} challenged the committee's previous conclusions.",
        f"{name} was acknowledged by the chair for their extensive work on the subject.",
        f"Following the recommendations by {name}, the committee decided to revise their initial position.",
        f"{name}'s previous work on the subject was highly regarded by the committee members.",
        f"During the testimony, {name} cited several recent studies to support their argument.",
        f"The historical context provided by {name} was essential for understanding the debate.",
        f"Representative Davis highlighted {name}'s contributions to the field during her speech.",
        f"{name}, a long-standing member of the board, offered a dissenting opinion.",
        f"The policy brief authored by {name} was circulated among the attendees.",
        f"In his memo, {name} outlined the potential impacts of the proposed changes.",
        f"{name} was called upon to further elaborate on their previous remarks.",
        f"The committee praised {name} for their meticulous research and presentation skills.",
        f"{name} responded to the queries regarding the accuracy of the data presented.",
        f"Several participants expressed their support for the initiatives suggested by {name}.",
        f"The legal perspectives shared by {name} were crucial to the discussions.",
        f"In their written submission, {name} addressed several critical issues.",
        f"{name}'s role as a mediator was commended by all parties involved.",
        f"The detailed analysis by {name} helped shape the committee's final decision.",
        f"During the recess, {name} discussed the implications of the hearing with the press.",
        f"The theoretical framework developed by {name} was a focal point of the session.",
        f"{name}'s credentials were verified by the committee before the hearing.",
        f"The cross-examination by {name} brought new facts to light.",
        f"{name} was responsible for compiling the comprehensive report on the issue.",
    ]

    # Count the occurrences where the name is recognized as the specified entity type
    count = 0
    total = 0
    for template in templates:
        doc = nlp(template)
        for ent in doc.ents:
            if ent.text == name and ent.label_ == entity_type:
                count += 1
        total += 1

    # Return the probability of the name being recognized as the specified entity type
    return count / total if total > 0 else 0


def match_committee_with_pol(row, congress):
    """
    This function serves to match the committee names with an existing house and senate database to correct any misspellings.
    """
    name = str(row["committee_member"])
    name = name.replace(".", "")
    year = int(row["year"])
    state = row["state_po"]
    # to prevent any errors in North/South,East/West
    if state == "NC" or state == "SC":
        state = "CAR"
    if state == "ND" or state == "SD":
        state = "DAK"
    if state == "VA" or state == "WV":
        state = "VIR"
    if state == "WA" or state == "DC":
        state = "WAS"

    if pd.isna(name) or name == "" or name == "." or name == "nan" or name is None:
        row["matched_name"] = "no_match"
        row["matched_party"] = "no_match"
        row["matched_office"] = "no_match"
        row["matched_year"] = "no_match"
        row["matched_method"] = "no_match"
        row["matched_score"] = "no_match"
        row["matched_state"] = "no_match"
        row["without_state_and_match"] = False
        return row

    # For a house member, range: (year - 2, year); for a senate member, range: (year - 6, year)
    house_range = range(year - 2, year + 1)
    senate_range = range(year - 6, year + 1)

    house_criteria = (congress["office"] == "house") & (
        congress["year"].isin(house_range)
    )
    senate_criteria = (congress["office"] == "senate") & (
        congress["year"].isin(senate_range)
    )

    congress_possible = congress[house_criteria | senate_criteria].copy()

    if len(congress_possible) == 0:
        print(f"No matches in {year}")
        row["matched_name"] = "no_match"
        row["matched_party"] = "no_match"
        row["matched_office"] = "no_match"
        row["matched_year"] = "no_match"
        row["matched_method"] = "no_match"
        row["matched_score"] = "no_match"
        return row

    # STEP 1: MATCH FULL NAME
    # Match the name with the congress_possible
    possible_names = congress_possible["candidate"].tolist()
    results = process.extract(name, possible_names)
    # Only keep results with score > 80
    results = [result for result in results if result[1] > 80]
    congress_match_fullname = congress_possible[
        congress_possible["candidate"].isin([result[0] for result in results])
    ]
    if pd.isna(state) and len(congress_match_fullname) >= 1:
        row["matched_name"] = congress_match_fullname["candidate"].values[0]
        row["matched_party"] = congress_match_fullname["party"].values[0]
        row["matched_office"] = congress_match_fullname["office"].values[0]
        row["matched_year"] = congress_match_fullname["year"].values[0]
        row["matched_method"] = "full_name"
        row["matched_score"] = results[0][1]
        row["matched_state"] = congress_match_fullname["state_po_original"].values[0]
        row["without_state_and_match"] = True
        return row

    else:
        # Check the state
        congress_match_fullname = congress_match_fullname[
            congress_match_fullname["state_po"] == state
        ]
        if len(congress_match_fullname) >= 1:
            row["matched_name"] = congress_match_fullname["candidate"].values[0]
            row["matched_party"] = congress_match_fullname["party"].values[0]
            row["matched_office"] = congress_match_fullname["office"].values[0]
            row["matched_year"] = congress_match_fullname["year"].values[0]
            row["matched_method"] = "full_name"
            row["matched_score"] = results[0][1]
            row["matched_state"] = congress_match_fullname["state_po_original"].values[
                0
            ]
            row["without_state_and_match"] = False
            return row

        # STEP 2 & 3: MATCH LAST NAME AND FIRST NAME IF POSSIBLE. IF NOT, ONLY MATCH LAST NAME
        else:
            last_name = row["last_name"]
            if pd.isna(last_name) or last_name == "" or last_name == ".":
                print(f"No matches: {name} in {year}")
                row["matched_name"] = "no_match"
                row["matched_party"] = "no_match"
                row["matched_office"] = "no_match"
                row["matched_year"] = "no_match"
                row["matched_method"] = "no_match"
                row["matched_score"] = "no_match"
                row["matched_state"] = "no_match"
                row["without_state_and_match"] = False
                return row

            try:
                first_name = row["committee_member"].replace(last_name, "").split()[0]
            except Exception:
                try:
                    first_name = row["committee_member"].split()[0]
                except Exception:
                    first_name = ""

            # Ensure 'last_name' is a list
            possible_last_names = congress_possible["last_name"].tolist()
            last_name_results = process.extract(last_name, possible_last_names)
            # Only keep results with score > 80
            last_name_results = [
                result for result in last_name_results if result[1] > 80
            ]
            congress_match_lastname = congress_possible[
                congress_possible["last_name"].isin(
                    [result[0] for result in last_name_results]
                )
            ]
            congress_match_lastname["first_name"] = ""
            for j, row_lastname in congress_match_lastname.iterrows():
                try:
                    congress_match_lastname.at[j, "first_name"] = (
                        row_lastname["candidate"]
                        .str.replace(row_lastname["last_name"], "")
                        .split()[0]
                    )
                except Exception:
                    try:
                        congress_match_lastname.at[j, "first_name"] = row_lastname[
                            "candidate"
                        ].split()[0]
                    except Exception:
                        continue

            congress_match_firstnamelist = congress_match_lastname[
                "first_name"
            ].tolist()
            first_name_results = (
                process.extract(first_name, congress_match_firstnamelist)
                if first_name != ""
                else None
            )
            if first_name_results is not None:
                first_name_results = [
                    result for result in first_name_results if result[1] > 40
                ]
                congress_match_lastandfirst = congress_match_lastname[
                    congress_match_lastname["first_name"].isin(
                        [result[0] for result in first_name_results]
                    )
                ]

            if (
                first_name_results is None
                or congress_match_firstnamelist is None
                or len(congress_match_firstnamelist) == 0
            ):
                # STEP 3: ONLY MATCH LAST NAME, MOST INACCRURATE
                if len(congress_match_lastname) >= 1:
                    if pd.isna(state):
                        row["matched_name"] = congress_match_lastname[
                            "candidate"
                        ].values[0]
                        row["matched_party"] = congress_match_lastname["party"].values[
                            0
                        ]
                        row["matched_office"] = congress_match_lastname[
                            "office"
                        ].values[0]
                        row["matched_year"] = congress_match_lastname["year"].values[0]
                        row["matched_method"] = "last_name_only"
                        row["matched_score"] = last_name_results[0][1]
                        row["matched_state"] = congress_match_lastname[
                            "state_po_original"
                        ].values[0]
                        row["without_state_and_match"] = True
                    else:
                        congress_match_lastname = congress_match_lastname[
                            congress_match_lastname["state_po"] == state
                        ]
                        if len(congress_match_lastname) >= 1:
                            row["matched_name"] = congress_match_lastname[
                                "candidate"
                            ].values[0]
                            row["matched_party"] = congress_match_lastname[
                                "party"
                            ].values[0]
                            row["matched_office"] = congress_match_lastname[
                                "office"
                            ].values[0]
                            row["matched_year"] = congress_match_lastname[
                                "year"
                            ].values[0]
                            row["matched_method"] = "last_name_only"
                            row["matched_score"] = last_name_results[0][1]
                            row["matched_state"] = congress_match_lastname[
                                "state_po_original"
                            ].values[0]
                            row["without_state_and_match"] = False
                        else:
                            print(f"No match: {name} in {year}")
                            row["matched_name"] = "no_match"
                            row["matched_party"] = "no_match"
                            row["matched_office"] = "no_match"
                            row["matched_year"] = "no_match"
                            row["matched_method"] = "no_match"
                            row["matched_score"] = "no_match"
                            row["matched_state"] = "no_match"
                            row["without_state_and_match"] = False
                else:
                    row["matched_name"] = "no_match"
                    row["matched_party"] = "no_match"
                    row["matched_office"] = "no_match"
                    row["matched_year"] = "no_match"
                    row["matched_method"] = "no_match"
                    row["matched_score"] = "no_match"
                    row["matched_state"] = "no_match"
                    row["without_state_and_match"] = False

            else:
                # STEP 2: MATCH LAST NAME AND THEN FIRST NAME, FIRST NAME SCORE > 40
                if len(congress_match_lastandfirst) >= 1:
                    if pd.isna(state):
                        row["matched_name"] = congress_match_lastandfirst[
                            "candidate"
                        ].values[0]
                        row["matched_party"] = congress_match_lastandfirst[
                            "party"
                        ].values[0]
                        row["matched_office"] = congress_match_lastandfirst[
                            "office"
                        ].values[0]
                        row["matched_year"] = congress_match_lastandfirst[
                            "year"
                        ].values[0]
                        row["matched_method"] = "last_and_first"
                        row["matched_score"] = last_name_results[0][1]
                        row["matched_state"] = congress_match_lastandfirst[
                            "state_po_original"
                        ].values[0]
                        row["without_state_and_match"] = True
                    else:
                        congress_match_lastandfirst = congress_match_lastandfirst[
                            congress_match_lastandfirst["state_po"] == state
                        ]
                        if len(congress_match_lastandfirst) >= 1:
                            row["matched_name"] = congress_match_lastandfirst[
                                "candidate"
                            ].values[0]
                            row["matched_party"] = congress_match_lastandfirst[
                                "party"
                            ].values[0]
                            row["matched_office"] = congress_match_lastandfirst[
                                "office"
                            ].values[0]
                            row["matched_year"] = congress_match_lastandfirst[
                                "year"
                            ].values[0]
                            row["matched_method"] = "last_and_first"
                            row["matched_score"] = last_name_results[0][1]
                            row["matched_state"] = congress_match_lastandfirst[
                                "state_po_original"
                            ].values[0]
                            row["without_state_and_match"] = False
                        else:
                            print(f"No match: {name} in {year}")
                            row["matched_name"] = "no_match"
                            row["matched_party"] = "no_match"
                            row["matched_office"] = "no_match"
                            row["matched_year"] = "no_match"
                            row["matched_method"] = "no_match"
                            row["matched_score"] = "no_match"
                            row["matched_state"] = "no_match"
                            row["without_state_and_match"] = False

        return row


def preprocess_dataframe(df, file_name):
    """
    This function serves to preprocess the dataframe for matching with the committee members.
    """

    df = df[df["file"] == file_name].copy()
    df["speaker_original"] = df["speaker"]
    df["section"] = df["section"].str.lower()

    prefix_list = [
        "mr",
        "mrs",
        "ms",
        "miss",
        "dr",
        "prof",
        "representative",
        "senator",
        "chairman",
        "chairwoman",
        "vice",
        "chair",
    ]

    # Create a regex pattern for prefixes, including case insensitivity and optional period
    prefix_pattern = r"(?i)^\b(" + "|".join(prefix_list) + r")[\.\s]*"

    # Replace prefixes in a case-insensitive manner
    df["speaker"] = df["speaker"].str.replace(prefix_pattern, "", regex=True)
    df["speaker"] = df["speaker"].str.lower().str.replace(r"\.", "", regex=True)

    return df


def ensure_columns_exist(df, columns):
    """
    This function serves to ensure that the columns exist in the dataframe.
    """

    for column in columns:
        if column not in df.columns:
            df[column] = None


def extract_chair_lastnames(speakers):
    """
    This function serves to extract the chair lastnames from the speakers list.
    """
    chair_list = [s for s in speakers if "chair" in s]
    chair_list = [s for s in chair_list if "the" not in s]
    chair_lastname_list = []
    if chair_list:
        for entry in chair_list:
            words = entry.split()
            for i, word in enumerate(words):
                if "chair" in word and i + 1 < len(words):
                    chair_lastname_list.append(words[i + 1])
    else:
        chair_lastname_list = []

    return chair_list, chair_lastname_list


def match_chair(dfh, dfc, chair_list, chair_lastname_list):
    """
    This function serves to match the chair with the committee members.
    """

    if not chair_lastname_list:
        dfc_temp = dfc[dfc["identity"].str.contains("chair", case=False)]
        dfc_temp = dfc_temp[~dfc_temp["identity"].str.contains("vice", case=False)]
        if len(dfc_temp) > 1:
            indices = dfh[
                (dfh["speaker"].str.contains("chair", case=False))
                & (~dfh["speaker"].str.contains("vice", case=False))
            ].index
            dfh.loc[indices, "matched"] = len(dfc_temp)
            dfh.loc[indices, "matched_name"] = dfc_temp["matched_name"].str.cat(
                sep=" & "
            )
            dfh.loc[
                indices,
                [
                    "matched_year",
                    "matched_state",
                    "matched_identity",
                    "matched_office",
                    "matched_method",
                    "matched_committee",
                ],
            ] = dfc_temp[
                [
                    "matched_year",
                    "matched_state",
                    "matched_identity",
                    "matched_office",
                    "matched_method",
                    "matched_committee",
                ]
            ].apply(lambda x: x.str.cat(sep=" & "))
        elif len(dfc_temp) == 1:
            indices = dfh[(dfh["speaker"].str.contains("chair", case=False))].index
            dfh.loc[indices, "matched"] = 1
            for col in [
                "matched_name",
                "matched_year",
                "matched_state",
                "matched_identity",
                "matched_office",
                "matched_method",
                "matched_committee",
            ]:
                dfh.loc[indices, col] = dfc_temp[col].values[0]

    else:
        match_lastnames(dfh, dfc, chair_lastname_list, "chair")


def extract_vice_chair_lastnames(speakers):
    """
    This function serves to extract the vice chair lastnames from the speakers list.
    """

    vice_chair_list = [s for s in speakers if "vice" in s]
    vice_chair_list = [s for s in vice_chair_list if "the" not in s]
    vice_chair_lastname_list = []
    for entry in vice_chair_list:
        words = entry.split()
        for i, word in enumerate(words):
            if "vice" in word and i + 1 < len(words):
                vice_chair_lastname_list.append(words[i + 1])
    return vice_chair_list, vice_chair_lastname_list


def match_vice_chair(dfh, dfc, vice_chair_list, vice_chair_lastname_list):
    """
    This function serves to match the vice chair with the committee members.
    """

    if not vice_chair_lastname_list:
        dfc_temp = dfc[dfc["identity"].str.contains("vice", case=False)]
        update_dataframe(dfh, dfc_temp, vice_chair_list, "vice-chair")
    else:
        match_lastnames(dfh, dfc, vice_chair_lastname_list, "vice-chair")


def update_dataframe(dfh, dfc_temp, speaker_list, role):
    if len(dfc_temp) > 1:
        update_multiple_matches(dfh, dfc_temp, speaker_list)
    elif len(dfc_temp) == 1:
        update_single_match(dfh, dfc_temp, speaker_list)
    else:
        update_no_match(dfh, speaker_list)


def update_multiple_matches(dfh, dfc_temp, speaker_list):
    indices = dfh[dfh["speaker"].isin(speaker_list)].index
    dfh.loc[indices, "matched"] = len(dfc_temp)
    for col in [
        "matched_name",
        "matched_year",
        "matched_state",
        "matched_identity",
        "matched_office",
        "matched_method",
        "matched_committee",
    ]:
        dfh.loc[indices, col] = dfc_temp[col].str.cat(sep=" & ")


def update_single_match(dfh, dfc_temp, speaker_list):
    indices = dfh[dfh["speaker"].isin(speaker_list)].index
    dfh.loc[indices, "matched"] = 1
    for col in [
        "matched_name",
        "matched_year",
        "matched_state",
        "matched_identity",
        "matched_office",
        "matched_method",
        "matched_committee",
    ]:
        dfh.loc[indices, col] = dfc_temp[col].values[0]


def update_no_match(dfh, speaker_list):
    indices = dfh[dfh["speaker"].isin(speaker_list)].index
    dfh.loc[indices, "matched"] = 0
    dfh.loc[
        indices,
        [
            "matched_name",
            "matched_year",
            "matched_state",
            "matched_office",
            "matched_method",
            "matched_committee",
        ],
    ] = "no_match"
    dfh.loc[indices, "matched_identity"] = "no_identity"


def match_lastnames(dfh, dfc, lastname_list, role):
    """
    This function serves to match the lastnames with the committee members.
    """

    dfc["last_name"] = dfc["last_name"].fillna("")
    dfc["matched_name"] = dfc["matched_name"].fillna("")
    for entry in lastname_list:
        result = process.extractOne(entry, dfc["matched_name"])
        if result and result[1] > 80:
            matched_row = dfc[dfc["matched_name"] == result[0]]
            update_single_match(dfh, matched_row, [entry])
        else:
            update_no_match(dfh, [entry])


def match_remaining_speakers(dfh, dfc, speakers, prefix_list):
    dfc["last_name"] = dfc["last_name"].fillna("")
    dfc["matched_name"] = dfc["matched_name"].fillna("")
    speaker_lastname_list = [
        s.replace(prefix, "").strip() for s in speakers for prefix in prefix_list
    ]
    potential_speakers = dfc["matched_name"].tolist()
    potential_speakers_lastname = [
        s.split()[-1] for s in potential_speakers if s
    ]  # Ensure non-empty strings

    for last_name in speaker_lastname_list:
        result = process.extractOne(last_name, potential_speakers_lastname)
        if result and result[1] > 80:
            matched_row = dfc[dfc["matched_name"].str.contains(result[0], na=False)]
            if len(matched_row) == 1:
                update_single_match(dfh, matched_row, [last_name])
            elif len(matched_row) > 1:
                update_multiple_matches(dfh, matched_row, [last_name])
            else:
                update_no_match(dfh, [last_name])
        else:
            update_no_match(dfh, [last_name])


def match_hearing_with_committee_member(tomatch, matched, file_name):
    """
    This is the main function to match the hearing with the committee members.
    """

    dfh = preprocess_dataframe(tomatch, file_name)
    dfc = matched[matched["file_name"] == file_name].copy()

    ensure_columns_exist(
        dfh,
        [
            "matched",
            "matched_name",
            "matched_year",
            "matched_state",
            "matched_identity",
            "matched_office",
            "matched_method",
            "matched_committee",
        ],
    )

    speakers = dfh["speaker"].unique().tolist()

    chair_list, chair_lastname_list = extract_chair_lastnames(speakers)
    match_chair(dfh, dfc, chair_list, chair_lastname_list)

    vice_chair_list, vice_chair_lastname_list = extract_vice_chair_lastnames(speakers)
    match_vice_chair(dfh, dfc, vice_chair_list, vice_chair_lastname_list)

    remaining_speakers = [
        s for s in speakers if s not in chair_list and s not in vice_chair_list
    ]
    remaining_speakers = [
        s for s in remaining_speakers if "chair" not in s and "vice" not in s
    ]
    prefix_list = ["mr", "mrs", "ms", "miss", "dr", "prof", "representative", "senator"]
    match_remaining_speakers(dfh, dfc, remaining_speakers, prefix_list)

    return dfh


def hearing_sentiment(df, pipe_model=pipe, batch_size=10000):
    """
    This function serves to estimate sentiment to the hearing text.
    """

    sentiments = []
    sentiment_scores = []

    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        text_list = df[start:end]["text"].tolist()

        results = pipe_model(text_list)

        sentiments.extend([result["label"] for result in results])
        sentiment_scores.extend([result["score"] for result in results])

    # Assign the results back to the dataframe
    df["sentiment"] = sentiments
    df["sentiment_score"] = sentiment_scores

    return df


def find_possible_ocr_typo(text, target, threshold=90):
    n = len(text)
    matches = []

    for i in range(n):
        for j in range(i + 1, n + 1):
            substring = text[i:j]
            score = fuzz.ratio(target, substring)
            if score >= threshold:
                matches.append((substring, score))

    matches = sorted(set(matches), key=lambda x: x[1], reverse=True)

    def is_mostly_uppercase(s):
        uppercase_count = sum(1 for c in s if c.isupper())
        return uppercase_count > len(s) / 2

    filtered_matches = [m[0] for m in matches if is_mostly_uppercase(m[0])]

    return filtered_matches


def update_ocr_df(df, file, threshold=90):
    df_temp = df[df["file"] == file].copy()
    target_list = df_temp["speaker"].unique().tolist()

    for i, row in df_temp.iterrows():
        output = {}
        text = row["text"]
        for target in target_list:
            matches = find_possible_ocr_typo(text, target, threshold)
            if matches:
                output[target] = matches[0]
        if output:
            df_temp.at[i, "ocr_corrections"] = output

    return df_temp


def parse_text(text_original):
    results = {}
    # Step 1: Deal with Prefix Errors
    text = text_original.lower().replace("  ", " ").replace("\n", " ")

    speaker_regex = re.compile(
        r"(mr\.|mrs\.|ms\.|miss|senator|representative|chairman|vice|dr\.)\s([^\s]+)"
    )
    # how to be more accurate in prefixes?
    speaker_matches = re.findall(speaker_regex, text)
    # print(speaker_matches)

    for prefix, candidate in speaker_matches:
        full_candidate = f"{prefix} {candidate}"
        # only consider about capital situation of candidate not prefix regarding "Senator XXXX" situation
        for match in re.finditer(re.escape(candidate), text_original, re.IGNORECASE):
            original_candidate = match.group(0)
            upper_count = sum(1 for c in original_candidate if c.isupper())
            if upper_count > len(original_candidate) * 0.5 and upper_count > 3:
                original_idx = match.start()
                results[full_candidate] = original_idx

    # Step 2: Extract Capitalized Words and compare with "CHAIRMAN"
    words = text_original.split()
    valid_words = []
    for word in words:
        if len(word) > 3:
            upper_count = sum(1 for c in word if c.isupper())
            if upper_count > len(word) * 0.7:
                valid_words.append(word)

    for word in valid_words:
        if "VICE" in word:
            results[f"vice-chair_{word}"] = text_original.find(word)
            valid_words.remove(word)
            break

        if "CHAIR" in word:
            results[f"chair_{word}"] = text_original.find(word)
            valid_words.remove(word)
            break

    # Irregular words
    for word in valid_words:
        scores_chair = [
            fuzz.ratio(word, "CHAIRMAN"),
            fuzz.ratio(word, "CHAIRWOMAN"),
            fuzz.ratio(word, "CHAIR"),
        ]

        if any(score > 70 for score in scores_chair):
            original_idx = text_original.find(word)
            results["chair"] = original_idx

        scores_vicechair = [fuzz.ratio(word, "VICECHAIR"), fuzz.ratio(word, "VICE")]

        if any(score > 70 for score in scores_vicechair):
            original_idx = text_original.find(word)
            results["vice-chair"] = original_idx

    results = dict(sorted(results.items(), key=lambda item: item[1]))

    # Step 3: Remove Duplicates (to specify, when two keys share a same value, remove the shorter one)
    to_remove = []

    for key, value in results.items():
        for k, v in results.items():
            if k != key and v == value:
                if len(k) > len(key):
                    to_remove.append(key)
                else:
                    to_remove.append(k)

    for key in set(to_remove):
        results.pop(key, None)

    return results


def line_change(row, text_original, results):
    if not results:
        row = row.copy()
        row["note"] = "no_error"
        return pd.DataFrame(row).T

    new_rows = []
    text_split = {}
    length = len(text_original)

    split_points = sorted(results.values())

    if split_points[0] != 0:
        split_points.insert(0, 0)

    if split_points[-1] != length:
        split_points.append(length)

    for i in range(len(split_points) - 1):
        start_idx = split_points[i]
        end_idx = split_points[i + 1]
        segment = text_original[start_idx:end_idx].strip()

        if start_idx == 0:
            speaker = row["speaker"]
        else:
            speaker = next(key for key, value in results.items() if value == start_idx)

        text_split[start_idx] = {"text": segment, "speaker": speaker}

    for idx, segment in text_split.items():
        if idx == 0:
            row_original = row.copy()
            row_original["text"] = segment["text"]
            row_original["note"] = "first_part"
            row_original["speaker"] = segment["speaker"]
            row_original["line_changes"] = results
            new_rows.append(row_original)

        else:
            row_new = row.copy()
            row_new["text"] = segment["text"]
            row_new["note"] = "latter_part"
            row_new["speaker"] = segment["speaker"]
            row_new["line_changes"] = results
            new_rows.append(row_new)

    new_df = None

    new_rows = [new_row for new_row in new_rows if new_row.notna().any()]
    for new_row in new_rows:
        new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)

    return new_df.reset_index(drop=True)


def fix_line_change(idx, df):
    row = df.loc[idx]
    text_original = row["text"]
    results = parse_text(text_original)
    return line_change(row, text_original, results)


# chairman_regex = f"({"|".join(chairman_ocr_errors})"
# re.sub(pattern, 'THE CHAIRMAN.', string)


def detect_mostly_capital_sentence(sentence):
    """
    Is mostly capital: over 60% of the characters are mostly capital and after removing the non-mostly capital words,
    # the sentence is still longer than 5 characters.
    """
    words = sentence.split()
    length = len(words)

    if length <= 5:
        return False

    else:
        upper_word_count = 0
        for word in words:
            capital_count = sum(1 for c in word if c.isupper())
            if capital_count > len(word) * 0.6:
                upper_word_count += 1

        if upper_word_count <= length * 0.6:
            return False

        else:
            return True


def split_text_into_sentences_spacy(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]

    return sentences


def split_text_into_sentences_re(text):
    sentence_endings = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s")
    sentences = sentence_endings.split(text)
    # cleaned_sentences = [re.sub(r'[^a-zA-Z\s]', '', sentence).strip() for sentence in sentences if sentence]
    return sentences


def detect_section_heading(text):
    results = []
    sentences = split_text_into_sentences_re(text)

    for sentence in sentences:
        if detect_mostly_capital_sentence(sentence):
            results.append(sentence)

    return results


def is_sentence(text, nlp):
    """
    This function serves to determine whether the text is a sentence.
    """

    doc = nlp(text)

    contains_verb = any(token.pos_ == "VERB" for token in doc)
    contains_subject = any(token.dep_ == "nsubj" for token in doc)
    contains_object = any(token.dep_ == "dobj" for token in doc)
    contains_punctuation = any(token.is_punct for token in doc)
    contains_proper_noun = any(token.pos_ == "PROPN" for token in doc)

    score = 0
    if contains_verb:
        score += 1
    if contains_subject:
        score += 1
    if contains_object:
        score += 1
    if contains_punctuation:
        score += 1
    if contains_proper_noun:
        score -= 1

    return score


def deal_with_multiple_matched(row):
    """
    This function serves to deal with multiple matches.
    """

    row = row.copy()
    number = int(row["matched"])
    if number == 0:
        return row

    matched_name = row["matched_name"]
    matched_year = row["matched_year"]
    matched_state = row["matched_state"]
    matched_identity = row["matched_identity"]
    matched_office = row["matched_office"]
    matched_method = row["matched_method"]
    matched_committee = row["matched_committee"]

    matched_name = matched_name if not pd.isna(matched_name) else ""
    matched_year = matched_year if not pd.isna(matched_year) else ""
    matched_state = matched_state if not pd.isna(matched_state) else ""
    matched_identity = matched_identity if not pd.isna(matched_identity) else ""
    matched_office = matched_office if not pd.isna(matched_office) else ""
    matched_method = matched_method if not pd.isna(matched_method) else ""
    matched_committee = matched_committee if not pd.isna(matched_committee) else ""

    if number == 1:
        if "no_match" in matched_name:
            row["matched"] = 0
            return row
        else:
            return row

    if matched_name == "" or matched_year == "" or matched_name == "no_match":
        row["matched"] = 0
        row["matched_name"] = "no_match"
        row["matched_year"] = "no_match"
        row["matched_state"] = "no_match"
        row["matched_identity"] = "no_identity"
        row["matched_office"] = "no_match"
        row["matched_method"] = "no_match"
        row["matched_committee"] = "no_match"
        return row

    # remaining situation: number >= 2
    if pd.isna(matched_state):
        matched_state = "notapplicable & " * number
        matched_state = re.sub(r" & $", "", matched_state)

    names = matched_name.split(" & ")
    years = matched_year.split(" & ")
    states = matched_state.split(" & ")
    identities = matched_identity.split(" & ")
    offices = matched_office.split(" & ")
    methods = matched_method.split(" & ")
    committees = matched_committee.split(" & ")

    if len(states) != number:
        matched_state = "notapplicable & " * number
        matched_state = re.sub(r" & $", "", matched_state)
        states = matched_state.split(" & ")

    # connections can only sort out no_match
    connections = []
    for i in range(number):
        connection = f"{names[i]} | {years[i]} | {states[i]} | {identities[i]} | {offices[i]} | {methods[i]} | {committees[i]}"
        connections.append(connection)

    connections = list(set(connections))

    connections = [x for x in connections if "no_match" not in x]

    length = len(connections)

    row["matched"] = length

    if length == 0:
        row["matched_name"] = "no_match"
        row["matched_year"] = "no_match"
        row["matched_state"] = "no_match"
        row["matched_identity"] = "no_identity"
        row["matched_office"] = "no_match"
        row["matched_method"] = "no_match"
        row["matched_committee"] = "no_match"
        return row

    if length == 1:  # multiple matches filtering "no_match" to only one
        single_connection = connections[0].split(" | ")
        row["matched_name"] = single_connection[0]
        row["matched_year"] = single_connection[1]
        row["matched_state"] = single_connection[2]
        row["matched_identity"] = single_connection[3]
        row["matched_office"] = single_connection[4]
        row["matched_method"] = single_connection[5]
        row["matched_committee"] = single_connection[6]
        return row

    if length >= 1:
        # deal with same person, different year or committee situation, depends only on name, state, office
        names = list(set(names))
        states = list(set(states))
        offices = list(set(offices))

        if len(names) == 1 and len(states) == 1 and len(offices) == 1:
            row["matched"] = 1
            row["matched_name"] = names[0]
            row["matched_year"] = years[0]
            row["matched_state"] = states[0]
            row["matched_identity"] = identities[0]
            row["matched_office"] = offices[0]
            row["matched_method"] = methods[0]
            row["matched_committee"] = committees[0]
            return row

        # deal with same person but recorded with nicknames situation
        if len(states) == 1 and len(offices) == 1:
            last_names = [name.split()[-1] for name in names]
            last_names = list(set(last_names))
            if len(last_names) == 1:
                row["matched"] = 1
                row["matched_name"] = names[0]
                row["matched_year"] = years[0]
                row["matched_state"] = states[0]
                row["matched_identity"] = identities[0]
                row["matched_office"] = offices[0]
                row["matched_method"] = methods[0]
                row["matched_committee"] = committees[0]
                return row

        # remaining situations: split up connections again to reorganize the row
        names = []
        years = []
        states = []
        identities = []
        offices = []
        methods = []
        committees = []

        for connection in connections:
            name, year, state, identity, office, method, committee = connection.split(
                " | "
            )
            names.append(name)
            years.append(year)
            states.append(state)
            identities.append(identity)
            offices.append(office)
            methods.append(method)
            committees.append(committee)

        row["matched_name"] = " & ".join(names)
        row["matched_year"] = " & ".join(years)
        row["matched_state"] = " & ".join(states)
        row["matched_identity"] = " & ".join(identities)
        row["matched_office"] = " & ".join(offices)
        row["matched_method"] = " & ".join(methods)
        row["matched_committee"] = " & ".join(committees)

        return row

    return row


def delete_undeleted_names(df):
    """
    This function serves to delete the undeleted names.
    """
    speaker_regex = re.compile(
        r"(mr\.|mrs\.|ms\.|miss|senator|representative|chairman|vice|dr\.)\s([^\s]+)",
        re.IGNORECASE,
    )

    for i in range(1, len(df)):
        name = df.at[i, "speaker"].lower()
        if "chair" in name:
            previous_text = df.at[i - 1, "text"].rstrip()
            current_text = df.at[i, "text"]
            current_text = re.sub(r"^\s*", "", current_text).lstrip()

            # Extract last word from previous_text and first word from current_text
            previous_last_word = previous_text.split()[-1] if previous_text else ""
            current_first_word = current_text.split()[0] if current_text else ""

            # Calculate fuzzy matching scores
            previous_score = fuzz.ratio(previous_last_word, "THE")
            first_scores = [
                fuzz.ratio(current_first_word, "CHAIRMAN"),
                fuzz.ratio(current_first_word, "CHAIRWOMAN"),
                fuzz.ratio(current_first_word, "CHAIR"),
            ]

            if previous_score > 70:
                previous_text = previous_text[: -len(previous_last_word)].rstrip()

            if any(score > 70 for score in first_scores):
                current_text = re.sub(
                    re.escape(current_first_word), "", current_text, flags=re.IGNORECASE
                ).lstrip()

            df.at[i - 1, "text"] = previous_text
            df.at[i, "text"] = current_text

        # Check if the 'note' column is 'latter_part'
        if df.at[i, "note"] == "latter_part":
            match = speaker_regex.match(name)

            if match:
                prefix = match.group(1)
                candidate = match.group(2)

                # Remove prefix from the end of the previous row's 'text' column
                previous_text = df.at[i - 1, "text"].rstrip()
                if previous_text.lower().endswith(prefix.lower()):
                    df.at[i - 1, "text"] = previous_text[: -len(prefix)].rstrip()

                # Remove candidate from the start of the current row's 'text' column
                current_text = df.at[i, "text"]
                candidate_pattern = re.compile(
                    r"^\s*" + re.escape(candidate), re.IGNORECASE
                )
                current_text = re.sub(candidate_pattern, "", current_text).lstrip()
                df.at[i, "text"] = current_text

    return df


def spell_check(text, spell):
    """
    This function serves to spell check the text.
    """
    if not isinstance(text, str):
        return 0, 0, 0

    text = re.sub(r"[^a-zA-Z\s\']", "", text).lower()
    words = text.split()
    unknown_words = spell.unknown(words)
    total_count = len(words)

    short_word_count = 0

    for word in words:
        if len(word) <= 1 and word != "i":
            short_word_count += 1

    return len(unknown_words), total_count, short_word_count

```

[Return to Sample Codes](../sample_codes.md)