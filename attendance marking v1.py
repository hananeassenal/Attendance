# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 23:35:50 2024

@author: Khadija
"""

import csv
import os

def mark_attendance(names, date, filename):
    # Check if the CSV file exists
    file_exists = os.path.isfile(filename)

    # Step 1: Create a set of existing names and dates
    existing_names = set()
    existing_dates = set()

    if file_exists:
        with open(filename, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                existing_names.add(row['names'])
                for fieldname in reader.fieldnames:
                    if fieldname != 'names':
                        existing_dates.add(fieldname)

    # Step 2: Add the specified date if it doesn't exist
    if date not in existing_dates:
        existing_dates.add(date)

        # Sort the existing dates
        existing_dates = sorted(existing_dates)

        # Update the CSV file with the new header
        with open(filename, 'r', newline='') as csvfile:
            data = list(csv.DictReader(csvfile))

        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['names'] + existing_dates
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            # Write the existing data back to the file
            writer.writerows(data)

    # Step 3: Mark attendance for each name under the specified date
    with open(filename, 'r', newline='') as csvfile:
        data = list(csv.DictReader(csvfile))

    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['names'] + sorted(existing_dates)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        for name in names:
            if name in existing_names:
                row = next(row for row in data if row['names'] == name)
                row[date] = '1'
            else:
                # If the name is not in the CSV, create a new row with the name
                row = {'names': name}
                for d in existing_dates:
                    row[d] = '0'  # Initialize other dates with '0'
                row[date] = '1'  # Mark attendance for the specified date
                data.append(row)

        writer.writeheader()
        writer.writerows(data)

# Example usage:
name_list = ['Eve', 'me','nishant']
attendance_date = '2023-10-22'
mark_attendance(name_list, attendance_date, 'names.csv')