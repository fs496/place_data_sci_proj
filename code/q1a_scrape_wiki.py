"""
Script for scraping the Wikipedia page "List of natural disasters by death toll"
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd


def parse_table(table) -> pd.DataFrame:
    """
    Parse a table tag to create a pandas DataFrame.
    
    This function includes special handling to deal with merged cells
    in the tables. The tables contain merged cells within a given column,
    resulting in values that must be forward-filled into subsequent rows.
    
    Parameters
    ----------
    table: bs4.element.Tag
        Tag that contains a table
    
    Returns
    -------
    pd.DataFrame
        A pandas Data.Frame containing the parsed table
    """
    columns = [tag.text for tag in table.find_all('th')]
    data = []
    rowspans = {j: 1 for j in range(0, len(columns))}
    previous_vals = {j: "" for j in range(0, len(columns))}
    for row in table.find_all('tr')[1:]:
        print("Working on a new row")
        # Set up a blank dictionary for the row
        row_data = {j: "" for j in range(0, len(columns))}
        # Check to see if we have any ongoing merged cells
        # from any columns from earlier rows
        insert_previous_vals = {
            j: True if rowspans[j] > 1 else False
            for j in range(0, len(columns))
        }
        # If there are, insert the saved values in the appropriate
        # columns and reduce the rowspan count to reflect that the
        # merged cell has been forward-filled
        for j, insert_val in insert_previous_vals.items():
            if insert_val:
                print(f"Inserting a previous val: col {j}, value {previous_vals[j]}")
                row_data[j] = previous_vals[j]
                rowspans[j] = rowspans[j] - 1
        
        # Now iterate through the data points for the current row
        for col_val in row.find_all('td'):
            # Find the first empty value of the row data
            # That's actual column that we are on
            j = [k for k, v in row_data.items() if v == ""][0]
            # Check if the current data point has a rowspan
            # value, indicating that this is the beginning of a new
            # merged cell
            try:
                rowsp = int(col_val['rowspan'])
                if rowsp > 1:
                    print(f"Found merged cell: col {j}, val {col_val.text}, rowspan {rowsp}")
                    rowspans[j] = rowsp
                    previous_vals[j] = col_val.text
            except KeyError:
                pass
            # Save the current data point in the right column
            row_data[j] = col_val.text
        data.append(row_data)
    df = pd.DataFrame(data=data)
    df.columns = columns
    return df


if __name__ == '__main__':
    # Get the HTML for the Wikipedia page - only ran this once to avoid
    # getting blocked
    url = "https://en.wikipedia.org/wiki/List_of_natural_disasters_by_death_toll"
    user_agent = 'School project (EMAIL REMOVED FOR PRIVACY)'
    headers = {'User-Agent': user_agent}
    page = requests.get(url, headers=headers)
    
    soup = BeautifulSoup(page.content, 'html.parser')
    # Find the 20th and 21st century deadliest disasters tables
    table_20 = soup.find_all('table', id='mwvA')[0]
    table_21 = soup.find_all('table', id='mwA9g')[0]

    df_20 = parse_table(table_20)    
    df_21 = parse_table(table_21)
    
    # Save tables
    save_folder = "(TOP FOLDER REMOVED FOR PRIVACY)/place_data_sci_proj/data"
    df_20.to_csv(f"{save_folder}/natural_disasters_20.csv", index=False)
    df_21.to_csv(f"{save_folder}/natural_disasters_21.csv", index=False)
