"""

SI 630 | WN 2020 | Ji Hwang (Henry) Kim

HW 0: Develop a code that uses  regular  expressions  to  extract  and
canonicalize email addresses from web page, then save them in csv format

"""

# Imports
import csv, re

# Initialize storage for emails
email_list = []

# Open the given file, then do:
with open("W20_webpages.txt", "r") as pages:

    # For each line in the file, do:
    for i, page in enumerate(pages):

        # Get rid of the fixed HTML format in the string, <html><head></head><body><p> and </p><br /></body></html>
        page = page[28:-25]

        # Replace @@ variation to @
        page = page.replace("@@", "@")

        # Find email address in the page in format: "id@domain.sth"
        result = re.search(r'[\w\`\~\!\#\$\%\^\&\*\(\)\-\_\=\+\[\]\{\}\\\|\;\:\'\"\/\?\,\.\<\>]+@[\w\`\~\!\#\$\%\^\&\*\(\)\-\_\=\+\[\]\{\}\\\|\;\:\'\"\/\?\,\.\<\>]+\.[\w\`\~\!\#\$\%\^\&\*\(\)\-\_\=\+\[\]\{\}\\\|\;\:\'\"\/\?\,\.\<\>]+', page)

        # If found, add to the email list in specified format
        if result:
            email_list.append([i, result.group(0)])

        # If not found, do:
        else:

            # Split the webpage into words
            words = page.split()

            # If there are more than 5 words (in the sense that it can contain "id @ domain . sth" in some format), do:
            if len(words) >= 5:

                # For each 5 sets of words, do:
                for j in range(len(words) - 4):
                    phrase = words[j:j + 5]

                    # If "@" and "." are the 2nd and 4th word respectively, found one!
                    if ("@" == phrase[1]) and ("." == phrase[3]):
                        email_list.append([i, phrase[0] + "@" + phrase[2] + "." + phrase[4]])
                        break

                    # Else, if "at" and "dot" are the 2nd and 4th word respectively, found one!
                    elif ("at" == phrase[1]) and ("dot" == phrase[3]):
                        email_list.append([i, phrase[0] + "@" + phrase[2] + "." + phrase[4]])
                        break

                    # Else, if "[at]" and "[dot]" are the 2nd and 4th word respectively, found one!
                    elif ("[at]" == phrase[1]) and ("[dot]" == phrase[3]):
                        email_list.append([i, phrase[0] + "@" + phrase[2] + "." + phrase[4]])
                        break

                    # Else, if "/at/" and "/dot/" are the 2nd and 4th word respectively, found one!
                    elif ("/at/" == phrase[1]) and ("/dot/" == phrase[3]):
                        email_list.append([i, phrase[0] + "@" + phrase[2] + "." + phrase[4]])
                        break

        # If nothing has been added to the email list from this page, do:
        if email_list[-1][0] != i:

            # If not found, add None
            email_list.append([i, "None"])

# Open the csv file to write to, then do:
with open("email-outputs.csv", "w", newline="") as f:

    # Initialize write for the file
    w = csv.writer(f)

    # Write initial row specified in Kaggle
    w.writerow(["Id", "Category"])

    # For each item in the list, do:
    for email in email_list:

        # Write to each row of the file
        w.writerow(email)