import numpy as np
import pandas as pd

def main(args):
    print('--- Reading wikitable of counties ---')
    # scrape county tables first
    county_df = pd.read_html('https://en.wikipedia.org/wiki/User:Michael_J/County_table', attrs={"class": "wikitable"})[0]

    print('--- Loading covid table counts ---')
    # now scrape current case count (updated daily)
    cases_df = pd.read_csv('https://usafactsstatic.blob.core.windows.net/public/data/covid-19/covid_confirmed_usafacts.csv', sep=',',
                        error_bad_lines=False)

    # utility functions for scraping
    def grab_identification_data(fips_code, name='Population(2010)'):
        # this one grabs the selected column name from county_df according to a fips code

        if fips_code != 0:
            county_location_data = county_df[county_df['FIPS'] == fips_code]
            county_location_keep = county_location_data[name]
            return np.array(county_location_keep)
        else:
            print('Warning (bad fips code): ', fips_code)
            mdict = {name: [np.nan]}
            county_location_keep = pd.DataFrame(mdict)
            return np.array(county_location_keep)[0][0]

    def series_to_arr(series):
        # converts a pd series to numpy array
        current_list = []
        for p in series.tolist():
            try:
                p[0];
                current_list.append(p.tolist()[0])
            except:
                current_list.append(p)
        return np.array(current_list)

    def coords_to_float(tmp):
        # converts latitude and longitude to floats
        floated_tmp = np.zeros_like(tmp)
        for i, t in enumerate(tmp):
            try:
                vals = t[:-1]
                if vals[0] == '–':
                    floated_tmp[i] = -1 * float(vals[1:-1])
                else:
                    floated_tmp[i] = float(vals[1:-1])
            except:
                continue
        return floated_tmp

    print('--- Grabbing county level features ---')
    # we now pull out the features
    population = cases_df['countyFIPS'].apply(grab_identification_data, name='Population(2010)')
    landarea = cases_df['countyFIPS'].apply(grab_identification_data, name='Land Areakm²')
    latitude = cases_df['countyFIPS'].apply(grab_identification_data, name='Latitude')
    longitude = cases_df['countyFIPS'].apply(grab_identification_data, name='Longitude')

    # and convert them to np arrays
    cases_df['population'] = series_to_arr(population)
    cases_df['landarea'] = series_to_arr(landarea)
    cases_df['latitude'] = coords_to_float(series_to_arr(latitude))
    cases_df['longitude'] = coords_to_float(series_to_arr(longitude))

    # selecting these columns to keep when converting the cases_df into long format
    id_cols = cases_df.columns[:4].tolist()
    id_cols.extend(cases_df.columns[-4:].tolist())

    print('--- melting data ---')
    # melt the cases_df into long format
    long_cases_df = pd.melt(cases_df, id_vars=id_cols, 
            value_vars=cases_df.columns[4:-4], var_name='date', value_name='totalCount')
    long_cases_df.rename(columns={'County Name':'countyName'}, inplace=True)

    # finally we write the file
    long_cases_df.to_csv(args.output, sep=',')

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-o', '--output', help='output file name', default=None)
    args = parser.parse_args()
    main(args)