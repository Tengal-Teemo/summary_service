import os
import itertools

from pandas import DataFrame

from sexiest_diffs import summary_from_files


persona_ceo = "A visionary and strategic CEO with two decades of experience in the tech industry. Leads a Fortune 500 company, driving digital transformation and sustainable growth. Charismatic public speaker, adept at navigating complex business landscapes, and passionate about fostering innovation. Champions diversity and inclusion in the workplace, known for inspirational leadership and ability to set long-term organizational direction."
persona_cfo = "A meticulous and conservative CFO with an MBA from a top business school and extensive experience in investment banking. Excels in financial forecasting, risk management, and cost optimization. Ensures regulatory compliance and effectively communicates complex financial data to stakeholders. Known for securing favorable funding for company initiatives and maintaining a strong focus on fiscal responsibility and shareholder value."
persona_engineer = "A talented and innovative Software Engineer specializing in full-stack development with expertise in cloud-native applications and DevOps practices. Passionate about clean code, test-driven development, and staying current with emerging technologies. Actively contributes to open-source projects and mentors junior developers. Possesses strong problem-solving skills and collaborates effectively across teams, making significant contributions to development projects."

def main():
    test_path = 'test_files'
    test_files = [file for file in os.listdir(test_path) if not os.path.isdir(os.path.join(test_path, file))]

    test_files_sorted = sorted(test_files, key=lambda x: x[0].lower())
    test_files_grouped = {key: list(group) for key, group in itertools.groupby(test_files_sorted, key=lambda x: x[0].lower())}

    personas = {'ceo': persona_ceo, 'cfo': persona_cfo, 'engineer': persona_engineer}

    outfile = os.path.join(test_path, 'Outputs/', 'results1.csv')
    if os.path.exists(outfile):
        raise FileExistsError('Don\'t overwrite results!')

    outdict = {'Input File': [], 'Output File': [], 'Prompt Tokens': [], 'Response Tokens': []}

    for persona in personas.keys():
        outdict.update({persona: []})

    for file_versions in test_files_grouped.values():
        file_versions.sort(key=lambda x: x[1].lower())
        for first, second in zip(file_versions, file_versions[1:]):
            persona_responses, prompt_tokens, response_tokens = summary_from_files(os.path.join(test_path, first), os.path.join(test_path, second), personas)
            outdict['Input File'].append(first)
            outdict['Output File'].append(second)
            outdict['Prompt Tokens'].append(prompt_tokens)
            outdict['Response Tokens'].append(response_tokens)
            for role, response in persona_responses.items():
                outdict[role].append(response)

    df = DataFrame.from_dict(outdict)
    df.to_csv(outfile)

if __name__ == '__main__':
    main()
