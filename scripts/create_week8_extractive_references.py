"""Freeze source-only reference drafts using independently chosen sentence anchors.

The anchors below were selected by the assistant after reading each original
remark, before running the revised selector on the new test cohort. This is a
draft annotation exercise, not teammate evaluation or model-generated gold.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.listing_summarizer import ListingSummarizer

DEV_ANCHORS = {
    '1179227225': ['Tucked away on a tree-lined street', 'The thoughtfully reimagined interior', 'Outside, the expansive'],
    '1174445030': ['Nestled within the sought-after enclave', 'The kitchen anchors the heart'],
    '1189490079': ['Tucked away in the foothills', 'The community features a sparkling pool', 'The In-Unit laundry room'],
    '1179199613': ['Located in the quiet desert community', 'Set on a spacious lot'],
    '1158795840': ['Set on a rare oversized bayfront lot', 'Inside, four bedrooms and four baths'],
    '1160004120': ['WELCOME TO THIS BEAUTIFUL', 'The property features a detached rear garage', 'The home has been recently updated'],
    '1176698043': ['energy efficient 2 bed 2 bath', 'The sparkling pool and spa', "You'll appreciate your energy savings"],
    '1157540545': ['Curved archways and textured walls', 'Three bedrooms and a private office', 'The sun shines across'],
    '1179231055': ['Spacious 3BR/2BA Garden Grove', 'Enjoy a functional single-story layout', 'The crown jewel of this property'],
    '1182868514': ['Move-in ready townhome-style condo', 'Upstairs, two spacious bedrooms', 'Additional features include an attached'],
    '1184344838': ['Welcome to 6050 E.', 'The stunning gourmet kitchen', 'An approximately one-year-old spa'],
    '1179325362': ['Located on a quiet cul-de-sac', 'A portion of the garage', 'The backyard is ready for family fun'],
    '1190863047': ['Gorgeous completely remodeled Mountain View', 'The custom kitchen anchors', 'Sliding doors lead to the private'],
    '1189566652': ['Welcome to this beautifully upgraded end-unit', 'Residents of Grandview Crest'],
    '1190507668': ['This stunning 4-bedroom', 'Nearby San Bruno Mountain'],
    '1179907859': ['Spanning an impressive 1,427', 'Enhance your living experience'],
    '1176324617': ['With timeless beach cottage charm', 'The rooftop deck captures'],
    '1176119202': ['Nestled in the highly desirable Scripps', 'Freshly painted throughout', 'Additional highlights include a detached'],
    '1190885923': ['Welcome to this charming and beautifully maintained condo', 'Outstanding parking setup', 'Enjoy top-notch community amenities'],
    '1190171882': ['4 beds and 2.5 baths', 'Soaring wood ceilings', "Down the hall you'll find"],
}

TEST_ANCHORS = {
    '1156538875': ['West Palmdale home with 3 bedrooms', 'The living room features', 'There is also a two-car detached garage'],
    '1189473380': ['This charming 4-bedroom', 'Situated on a spacious lot', 'An attached garage adds'],
    '1176603062': ['Set on a generous 7,613', 'Currently configured with two bedrooms', 'Just moments from Toluca Lake'],
    # The source has one concise factual sentence followed by an unpunctuated
    # marketing/list section. Keep the single sentence rather than inventing text.
    '1174225864': ['Located in the highly sought-after Temecula'],
    '1174744255': ['BRAND NEW WHITE CAESARSTONE', 'This seamlessly connects to a huge open living room', 'There also is a gorgeous swimming pool'],
    '1178234327': ['PRICE IMPROVEMENT', 'Located in one of the most desirable school districts', 'The all-new kitchen features'],
    '1183472536': ['Welcome to this beautifully maintained single-story pool home', 'Built in 2009 and situated', 'Step outside to a beautifully landscaped backyard'],
    '1183147905': ['Panoramic ocean views.', 'This professionally designed coastal retreat', 'From the moment you arrive'],
    '1176286746': ['Welcome to 20948 Center St', 'Perfectly positioned on an expansive', 'Step outside to your own private backyard'],
    '1176952573': ['The home is newly remodeled', 'A very rare find 3 bedroom', 'There are two patios'],
    '1157825716': ['Original 1940 two-story Idyllwild', 'Off one side of the kitchen', 'Off the other side of the kitchen'],
    '1159976751': ['Two great homes on one lot', 'The main home offers 3 bedrooms', 'This 1 bedroom, 1 bathroom home'],
    '1179307732': ['This well-maintained home features OWNED SOLAR', 'All three bedrooms', 'The low-maintenance backyard'],
    '1190347015': ['Savor PANORAMIC MOUNTAIN VIEWS', 'PAID OFF SOLAR', 'An elegant arch connects'],
    '1177002563': ['40500 Via Tapadero, Murrieta', 'Built around a scenic 18-hole golf course', 'Bathroom requires repair'],
    '1189332060': ['Discover this stunning single-story home', 'The inviting living room', 'Your private backyard oasis'],
    '1180290080': ['Welcome to this beautifully refreshed end-unit', 'An oversized attached 2-car garage', 'Residents also enjoy access'],
    '1178544808': ['Set within the serene Streamhouse', 'Lovingly maintained by only two owners', 'Residents of Streamhouse enjoy'],
    '1159558215': ['Welcome to Villa Colina', 'Beautifully remodeled and move-in ready', 'The oversized 2-car garage'],
    '1176989295': ['Welcome to this charming home situated', 'Full of potential, this home', 'The rectangular lot offers'],
}


def build(split, output):
    if output.exists():
        raise FileExistsError(f'Keep existing approvals and references: {output}')
    legacy = json.loads((ROOT / 'tests/fixtures/week8_summary_references.json').read_text())['cases']
    if split == 'development':
        records = [case['record'] for case in legacy]
        anchors = DEV_ANCHORS
        seed = 8
    else:
        legacy_test = json.loads((ROOT / 'tests/fixtures/week8_test_references.json').read_text())['cases']
        excluded = {case['listing_id'] for case in legacy + legacy_test}
        frame = pd.read_csv(ROOT / 'data/processed/listing_sample.csv')
        records = frame[~frame.L_ListingID.astype(str).isin(excluded)].sample(n=20, random_state=20261115).to_dict('records')
        anchors = TEST_ANCHORS
        seed = 20261115
    tokenizer = ListingSummarizer()
    cases = []
    for record in records:
        listing_id = str(record['L_ListingID'])
        sentences = tokenizer.split_sentences(record['remarks'])
        indices = []
        for anchor in anchors[listing_id]:
            matches = [i for i, sentence in enumerate(sentences) if anchor.lower() in sentence.lower()]
            if len(matches) != 1:
                raise ValueError(f'{listing_id}: anchor {anchor!r} matched {matches}')
            indices.append(matches[0])
        indices = sorted(set(indices))
        chosen = [sentences[i] for i in indices]
        cases.append({'listing_id': listing_id, 'record': record, 'split': split,
                      'reference_summary': ' '.join(chosen), 'reference_sentences': chosen,
                      'reference_sentence_indices': indices, 'source_only': True,
                      'source_sha256': hashlib.sha256(record['remarks'].encode()).hexdigest(),
                      'reference_source': 'assistant_selected_original_sentences',
                      'review_status': 'needs_human_review'})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'protocol': 'remarks-only independently selected sentence references; human confirmation pending',
                                 'random_seed': seed, 'excluded_previous_test': split == 'test',
                                 'cases': cases}, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Frozen {len(cases)} {split} source-only reference drafts: {output}')


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('split', choices=['development', 'test'])
    args = cli.parse_args()
    build(args.split, ROOT / f'tests/fixtures/week8_extractive_{args.split}.json')
