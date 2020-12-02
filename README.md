# Latent Dirichlet allocation for identifying latent fan types

Make sure to install the packages in `requirements.txt`.

To train the model and analyze the results, run the notebook `dp_user_groups.ipynb`. Note that it assumes access to the Snowflake data warehouse, and will have to be significantly amended to work with something else.

It assumes access to the following table: `user_artist_streams_unique_artist_sample`, with `columns` `artistname`, `canonical_artistid` (just a unique artist ID), `userid`, and `total_streams`. Each row is the total number of times a given listener (user) has streamed a given artist.
