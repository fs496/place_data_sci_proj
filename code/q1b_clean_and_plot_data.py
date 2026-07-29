"""
Clean and plot the natural disasters data.
"""
import pandas as pd
import re
import seaborn.objects as so


SAVE_FOLDER = "(TOP FOLDER REMOVED FOR PRIVACY)"
DATA_FOLDER = f"{SAVE_FOLDER}/data"


def average_ranges(num_range: str) -> float:
    """
    Average ranges to their midpoints.
    
    Parameters
    ----------
    num_range : str
        A range described by a string with the upper and lower bounds
        separated by a dash (–). May also be a string of a single number.

    Returns
    -------
    float
        Midpoint of the range, or the original number if only one is provided.
    """
    if '–' in num_range:
        bounds = re.split('–', num_range)
        assert len(bounds) == 2
        bounds = [int(b) for b in bounds]
        midpoint = (bounds[0] + bounds[1]) / 2
    else:
        midpoint = int(num_range)
    return midpoint


if __name__ == '__main__':
    # Read in the raw data scraped from Wikipedia
    df_20 = pd.read_csv(f"{DATA_FOLDER}/natural_disasters_20.csv")
    df_21 = pd.read_csv(f"{DATA_FOLDER}/natural_disasters_21.csv")
    
    # Concatenate 20th and 21st century data frames
    df_21 = df_21.rename(columns={'Death toll': 'Deaths'})
    df = pd.concat([df_20, df_21])
    df = df.reset_index(drop=True)
    
    # Drop a null row
    df = df.dropna(how='all', axis=0)
    
    # Make column names easier to work with
    df = df.rename(columns={'Countries affected': 'countries'})
    df = df.rename(str.lower, axis='columns')
    
    # Clean year column
    df.year = df.year.astype(int)
    
    # Clean deaths column
    df.deaths = df.deaths.str.replace("\[.*\]", "", regex=True)\
        .str.replace("+", "")\
        .str.replace(",", "")
    df.deaths = df.deaths.apply(average_ranges)
    
    # Clean disaster type column
    df.type = df.type.replace(to_replace='Floods', value='Flood')
    # According to the CDC, a mudslide is a type of landslide - rename
    # just for simplicity
    # See https://www.cdc.gov/landslides-and-mudslides/about/index.html
    df.type = df.type.str.replace("Mudslide", "Landslide")
    
    # Plot death toll versus year for each disaster, color coded by
    # type of disaster
    # Every year has only one disaster in the data (the deadliest that year),
    # except for 1972 which includes both the Qir earthquake (5,374 deaths)
    # and the Managua earthquake (4,000-11,000 deaths) - possibly because it
    # is unclear which was deadlier
    
    # Want to label some of the deadliest events
    df_plot = df.copy(deep=True)
    df_plot.loc[df_plot.deaths <= 250000, 'event'] = ''
    fig1 = (
        so.Plot(df_plot, x="year", y="deaths", color="type", text='event')
        .add(so.Dot())
        .add(so.Text(halign='right', valign='bottom', offset=2, color='black', fontsize=6.5))
        .label(title = """Deadliest natural disaster each year by death toll, 1901-2026
               Events with over 250,000 deaths labelled""")
        .label(x='Year', y='Number of deaths', color='Disaster type')
        .limit(x=(1890, None))
    )
    fig1.save(f"{SAVE_FOLDER}/figures/fig1.png", dpi=400, bbox_inches = "tight")
    
    # Now with log scale axes to better see the spread of the data
    fig2 = (
        so.Plot(df_plot, x="year", y="deaths", color="type")
        .add(so.Dot())
        .scale(y='log')
        .label(title = "Deadliest natural disaster each year by death toll, 1901-2026 - Log Scale")
        .label(x='Year', y='Number of deaths', color='Disaster type')
    )
    fig2.save(f"{SAVE_FOLDER}/figures/fig2.png", dpi=400, bbox_inches='tight')
