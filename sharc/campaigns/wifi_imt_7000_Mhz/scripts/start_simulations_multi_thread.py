from sharc.run_multiple_campaigns_mut_thread import run_campaign

# Set the campaign name
# The name of the campaign to run. This should match the name of the campaign directory.
name_campaign = "wifi_imt_7000_Mhz"

# Run the campaigns
# This function will execute the campaign with the given name.
# It will look for the campaign directory under the specified name and start the necessary processes.
run_campaign(name_campaign)
