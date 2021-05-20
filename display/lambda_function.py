import logging
import math
from convert_rasters import convert_granules


logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Define a list of Python lambda functions that are called by this AWS Lambda function.
ACTIONS = {
    'convert_granules': convert_granules
}


def lambda_handler(event, context):
    """
    Accepts an action and a number, performs the specified action on the number,
    and returns the result.
    :param event: The event dict that contains the parameters sent when the function
                  is invoked.
    :param context: The context in which the function is called.
    :return: The result of the specified action.
    """
    logger.info(f'Event: {event}')

    result = ACTIONS[event['action']](event['granule_list'], event['dest_dir'])
    logger.info('Calculated result:', result)

    response = {'result': result}
    return response
